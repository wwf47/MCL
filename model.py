import random

import torch

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
import copy
import math

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, T5Config, MT5ForConditionalGeneration, BartForConditionalGeneration, MBartTokenizer, MT5Tokenizer, BartTokenizer, MBartForConditionalGeneration, MT5Config, BartConfig, MBartConfig
from transformers import get_linear_schedule_with_warmup
from data_utils import ABSADataset, read_pola_from_file
from eval_utils import extract_spans_extraction, compute_f1_scores


class EMA(nn.Module):
    def __init__(self, model, decay, total_step=3000):
        super().__init__()
        self.decay = decay
        self.total_step = total_step
        self.step = 0
        self.model = copy.deepcopy(model).eval()

    def update(self, model):
        self.step = self.step+1
        decay_new = 1-(1-self.decay)*(math.cos(math.pi*self.step/self.total_step)+1)/2
        # decay_new = self.decay
        with torch.no_grad():
            e_std = self.model.state_dict().values()
            m_std = model.state_dict().values()
            for e, m in zip(e_std, m_std):
                e.copy_(decay_new * e + (1. - decay_new) * m)


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type=type_path, 
                       task=args.task, max_len=args.max_seq_length)

def get_polarity(data_type, task):
    pos = read_pola_from_file(f'data/{task}/{data_type}/pos.txt')
    neg = read_pola_from_file(f'data/{task}/{data_type}/neg.txt')
    neu = read_pola_from_file(f'data/{task}/{data_type}/neu.txt')
    return pos, neg, neu

sentiment_word_list = ['positive', 'negative', 'neutral']
aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options',
                    'food general']


class T5Pooler(nn.Module):
    def __init__(self, hidden_size):
        super(T5Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp_layers = 2
        self.mlp = nn.Sequential()
        for i in range(self.mlp_layers-1):
            self.mlp.add_module("mlp_"+str(i),nn.Linear(config.d_model, config.d_model))
            self.mlp.add_module("layer_norm_"+str(i),nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.mlp.add_module("dropout_"+str(i),nn.Dropout(config.dropout_rate))
        
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.Tanh()

    def forward(self, x, **kwargs):
        if self.mlp_layers > 1:
            x = self.mlp(x)
        
        x = self.dense(x)
        x = self.activation(x)
        return x


class ProjectionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj_layers = 1
        self.proj = nn.Sequential()

        for i in range(self.proj_layers - 1):
            self.proj.add_module("dense_" + str(i), nn.Linear(config.d_model, config.d_model))
            self.proj.add_module("layer_norm_" + str(i), nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.proj.add_module("dropout_" + str(i), nn.Dropout(config.dropout_rate))

        self.dense = nn.Linear(config.d_model, config.d_model)
        self.LayerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, **kwargs):
        if self.proj_layers > 1:
            x = self.proj(x)

        if self.proj_layers > 0:
            x = self.dense(x)
            x = self.LayerNorm(x)
            x = self.dropout(x)
        return x

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, last_hidden, hidden_states):

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.decay = hparams.ema_decay
        if "mt5" in hparams.model_name_or_path:
            self.model = MT5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
            self.config = MT5Config.from_pretrained(hparams.model_name_or_path)
        elif "t5" in hparams.model_name_or_path:
            self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
            self.config = T5Config.from_pretrained(hparams.model_name_or_path)
        elif "mbart" in hparams.model_name_or_path:
            self.model = MBartForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.tokenizer = MBartTokenizer.from_pretrained(hparams.model_name_or_path)
            self.config = MBartConfig.from_pretrained(hparams.model_name_or_path)
        elif "bart" in hparams.model_name_or_path:
            self.model = BartForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.tokenizer = BartTokenizer.from_pretrained(hparams.model_name_or_path)
            self.config = BartConfig.from_pretrained(hparams.model_name_or_path)
        else:
            print("there are something wrong with model")

        self.encoder = self.model.get_encoder()
        self.d_model = self.config.d_model
        self.project1 = nn.Sequential(nn.Linear(self.d_model, self.d_model*4), nn.ReLU(), nn.Linear(self.d_model*4, self.d_model))
        #self.project2 = nn.Sequential(nn.Linear(self.d_model, self.d_model*4), nn.ReLU(), nn.Linear(self.d_model*4, self.d_model))

        #self.projection = ProjectionLayer(self.config)
        self.pooler = Pooler(hparams.pool_type)
        #self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.con_loss = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.gen_cos = nn.CosineSimilarity(dim=-1)
        self.dropout = nn.Dropout(hparams.dropt)
        self.k = self.hparams.k
        self.T = self.hparams.T
        self.start = self.hparams.start_epoch
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
        self.senti_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
        #self.cross_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.config.num_heads, dropout=self.config.dropout_rate, batch_first=True)
        #contrastive learning
        self.tcl = hparams.tcl
        self.scl = hparams.scl
        self.element = hparams.element
        self.my_device = hparams.device
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        #self.w1 = nn.Parameter(torch.tensor(0.1))  # 定义为可学习的参数

    def is_logger(self):
        return True

    def get_cos(self, candi_emd, gen_emd):
        scores = []
        for c in candi_emd:
            score = self.gen_cos(c, gen_emd)
            scores.append(score)
        ranked_candi = sorted(candi_emd, key=lambda x: scores[x], reverse=True)
        return ranked_candi[0]


    def generate_contrast(self, candi, candi_mask, target):
        new_candi = []
        for i in range(self.hparams.train_batch_size):
            sam_c = candi[i:i+5,:]
            sam_mask = candi_mask[i:i+5,:]
            sam_t = target[i]
            sam_c_emd = self.encoder(sam_c, sam_mask).last_hidden_state
            print(sam_c_emd.size())
            new = self.gen_cos(sam_c_emd, sam_t)
            new_candi.append(new)
        new_candi = torch.tensor([item.cpu().detach().numpy() for item in new_candi])

        return new_candi

    def dynamic_weight_adjustment(self, loss, thresh=1.0):
        '''final_loss = self.w1*loss
        if final_loss > thresh and self.w1>=0.4:
            print("tri_loss is: " +str(loss.item())+"weight is" +str(self.w1))
            self.w1 *=0.9
        else:
            pass'''
        final_loss = self.w1 * loss
        tcl_grad_norm = self.project1[0].weight.grad.norm().item()
        print("grad is: " + str(tcl_grad_norm) + "tri_loss is: " + str(final_loss.item()) + "weight is" + str(self.w1.data))
        if tcl_grad_norm > thresh:
            # 根据需要，调整权重
            self.w1.data *= 1.1

        return final_loss

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None, cl=False):
        neg_cl = decoder_input_ids.clone()
        pos_cl = decoder_input_ids.clone()
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        loss = output[0]

        replace_cl = random.choice(["tcl", "scl"])

        if replace_cl == "tcl":
            if cl and self.tcl:
                ini_out = output.encoder_last_hidden_state
                avg_ini = torch.mean(ini_out, dim=1)
                norm_ini = F.normalize(avg_ini, dim=-1)
                ini_emd = self.project1(norm_ini)

                pos_out = self.encoder(pos_cl, decoder_attention_mask).last_hidden_state
                avg_pos = torch.mean(pos_out, dim=1)
                norm_pos = F.normalize(avg_pos, dim=-1)
                pos_emd = self.project1(norm_pos)

                replace_item = random.choice(["aspect", "opinion", "combine"])
                if self.hparams.task == "aste":
                    # random
                    if replace_item == "aspect":
                        neg_emd = self.multi_neg(neg_cl, input_ids, 'aspect')
                    elif replace_item == "opinion":
                        neg_emd = self.multi_neg(neg_cl, input_ids, 'opinion')
                    else:  # replace_item == "opinion"
                        neg_emd = self.multi_neg(neg_cl, input_ids, 'combine')
                elif self.hparams.task == "acsd":
                    replace_item = random.choice(["aspect", "cate", "combine"])

                    if replace_item == "aspect":
                        neg_emd = self.multi_neg(neg_cl, input_ids, 'aspect')
                    elif replace_item == "cate":
                        neg_emd = self.multi_neg(neg_cl, input_ids, 'cate')
                    else:
                        neg_emd = self.multi_neg(neg_cl, input_ids, 'combine_cate')

                else:
                    print("there are something wrong with task")

                if self.hparams.k == 0:
                    tri_loss = 1 - self.cos(ini_emd, pos_emd)
                elif self.hparams.k == 1:
                    tri_loss = self.triplet_loss(ini_emd, pos_emd, neg_emd)
                else:
                    pos_num = torch.exp(self.cos(ini_emd, pos_emd) / self.T)
                    neg_num = torch.exp(self.cos(ini_emd, neg_emd) / self.T)
                    neg_num_temp = neg_num.clone()
                    for i in range(self.hparams.k - 1):

                        if self.hparams.task == "aste":
                            replace_item = random.choice(["aspect", "opinion", "combine"])
                            # random
                            if replace_item == "aspect":
                                neg1_emd = self.multi_neg(neg_cl, input_ids, 'aspect')
                            elif replace_item == "opinion":
                                neg1_emd = self.multi_neg(neg_cl, input_ids, 'opinion')
                            else:  # replace_item == "opinion"
                                neg1_emd = self.multi_neg(neg_cl, input_ids, 'combine')
                        elif self.hparams.task == "acsd":
                            replace_item = random.choice(["aspect", "cate", "combine"])

                            if replace_item == "aspect":
                                neg1_emd = self.multi_neg(neg_cl, input_ids, 'aspect')
                            elif replace_item == "cate":
                                neg1_emd = self.multi_neg(neg_cl, input_ids, 'cate')
                            else:
                                neg1_emd = self.multi_neg(neg_cl, input_ids, 'combine_cate')

                        else:
                            print("there are something wrong with task")
                        neg1_num = torch.exp(self.cos(ini_emd, neg1_emd) / self.T)
                        neg_num_temp += neg1_num

                    neg_num = neg_num + neg_num_temp
                    tri_loss = -torch.log(pos_num / (pos_num + neg_num + 1e-8))

                # tri_loss = self.dynamic_weight_adjustment(tri_loss)

                loss += self.hparams.tcl_weight * torch.mean(tri_loss)
                # best_candi = self.generate_contrast(gen_ids, gen_mask, pos_out)
                # best_candis = torch.tensor(best_candi)
                # loss = self.loss_fct(best_candis.view(-1, best_candis.size(-1)), labels.view(-1))
        else:
            if cl and self.scl:
                pola_token, pola_mask, pp_token, pp_mask, pn_token, pn_mask = self.senti_token(input_ids.clone(),
                                                                                               pos_cl.clone())
                p_ini = self.multi_pola(pola_token, pola_mask)
                p_pos = self.multi_pola(pp_token, pp_mask)
                p_neg = self.multi_pola(pn_token, pn_mask)

                if self.hparams.k == 0:
                    senti_loss = 1 - self.cos(p_ini, p_pos)
                elif self.hparams.k == 1:
                    senti_loss = self.senti_loss(p_ini, p_pos, p_neg)
                else:
                    senti_pos_num = torch.exp(self.cos(p_ini, p_pos) / self.T)
                    senti_neg_num = torch.exp(self.cos(p_ini, p_neg) / self.T)
                    senti_neg_num_temp = senti_neg_num.clone()

                    for i in range(self.hparams.k - 1):
                        p1_token, p1_mask = self.get_neg_scl(input_ids.clone(), pos_cl.clone())
                        p1_neg = self.multi_pola(p1_token, p1_mask)
                        senti_neg1_num = torch.exp(self.cos(p_ini, p1_neg) / self.T)
                        senti_neg_num_temp += senti_neg1_num

                    senti_neg_num = senti_neg_num + senti_neg_num_temp

                    senti_loss = -torch.log(senti_pos_num / (senti_pos_num + senti_neg_num + 1e-8))

                loss += self.hparams.scl_weight * torch.mean(senti_loss)

        return loss

    def multi_neg(self, neg_token, input_token, ele):
        neg_token, neg_mask = self.get_neg_token(neg_token, input_token, ele)
        neg_out = self.encoder(neg_token.to(self.my_device), neg_mask.to(self.my_device)).last_hidden_state
        avg_neg = torch.mean(neg_out, dim=1)
        norm_neg = F.normalize(avg_neg, dim=-1)
        neg_emd = self.project1(norm_neg)
        return neg_emd

    def multi_pola(self, input_token, input_ma):
        out = self.encoder(input_token.to(self.my_device), input_ma.to(self.my_device)).last_hidden_state
        avg_pola = torch.mean(out, dim=1)
        norm_pola = F.normalize(avg_pola, dim=-1)
        pola_emd = self.project1(norm_pola)
        return pola_emd

    def get_pola(self, ini, dic):
        randid = random.randint(0, len(dic) - 1)
        if dic[randid].split(';')[1].strip() in ini:
            randid = random.randint(0, len(dic) - 1)
        pos_tt = dic[randid]
        return pos_tt

    def get_neg_scl(self, ini, token):
        targets = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in token]
        inits = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in ini]
        neg_token = torch.zeros((1, self.hparams.max_seq_length), dtype=torch.long, device=f'cuda')
        neg_mask = torch.zeros((1, self.hparams.max_seq_length), dtype=torch.long, device=f'cuda')
        pos, neg, neu = get_polarity(self.hparams.dataset, self.hparams.task)
        neg_t = ''
        for i in range(token.size(0)):
            tar_txt = targets[i]
            ini_txt = inits[i]

            if ';' not in tar_txt:
                if self.completed(tar_txt, 2):
                    a, o, c = tar_txt.split(', ')[:3]
                    if c == 'positive':
                        neg_t = self.get_pola(a, neg)
                    elif c == 'negative':
                        neg_t = self.get_pola(a, neu)
                    else:
                        neg_t = self.get_pola(a, pos)
                else:
                    neg_t = ini_txt
                    print("not completed")
                neg_text = self.tokenizer.batch_encode_plus([neg_t], max_length=self.hparams.max_seq_length,
                                                            pad_to_max_length=True, truncation=True,
                                                            return_tensors="pt")

                neg_input = neg_text['input_ids']
                neg_ma = neg_text['attention_mask']
            else:
                tri = tar_txt.split('; ')
                j = 0
                for ti in tri:
                    if self.completed(ti, 2):
                        a, o, c = ti.split(', ')[:3]
                        if c == 'positive':
                            neg_t = self.get_pola(a, neg)
                        elif c == 'negative':
                            neg_t = self.get_pola(a, neu)
                        else:
                            neg_t = self.get_pola(a, pos)
                    else:
                        neg_t = ini_txt
                        print("not completed")
                    neg_text = self.tokenizer.batch_encode_plus([neg_t], max_length=self.hparams.max_seq_length,
                                                                pad_to_max_length=True, truncation=True,
                                                                return_tensors="pt")
                    if j == 0:
                        neg_input = neg_text['input_ids']
                        neg_ma = neg_text['attention_mask']
                    else:
                        neg_input = torch.cat((neg_input, neg_text['input_ids']), 0)
                        neg_ma = torch.cat((neg_ma, neg_text['attention_mask']), 0)
                    j += 1
            if i == 0:
                neg_token = neg_input
                neg_mask = neg_ma
            else:
                neg_token = torch.cat((neg_token, neg_input), 0)
                neg_mask = torch.cat((neg_mask, neg_ma), 0)

        return neg_token, neg_mask

    def senti_token(self, ini, token):
        targets = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in token]
        inits = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in ini]
        n_batch = token.size(0)
        n_token = torch.zeros((1, self.hparams.max_seq_length), dtype=torch.long, device=f'cuda')
        n_mask = torch.zeros((1, self.hparams.max_seq_length), dtype=torch.long, device=f'cuda')
        pos_token = torch.zeros((1, self.hparams.max_seq_length), dtype=torch.long, device=f'cuda')
        pos_mask = torch.zeros((1, self.hparams.max_seq_length), dtype=torch.long, device=f'cuda')
        neg_token = torch.zeros((1, self.hparams.max_seq_length), dtype=torch.long, device=f'cuda')
        neg_mask = torch.zeros((1, self.hparams.max_seq_length), dtype=torch.long, device=f'cuda')
        pos, neg, neu = get_polarity(self.hparams.dataset, self.hparams.task)
        ini_pola = ''
        pos_t = ''
        neg_t = ''
        neg1_t = ''
        for i in range(n_batch):
            tar_txt = targets[i]
            ini_txt = inits[i]

            if ';' not in tar_txt:
                if self.completed(tar_txt, 2):
                    a, o, c = tar_txt.split(', ')[:3]
                    ini_pola = a + '; ' + o
                    if c == 'positive':
                        pos_t = self.get_pola(a, pos)
                        neg_t = self.get_pola(a, neg)

                    elif c == 'negative':
                        pos_t = self.get_pola(a, neg)
                        neg_t = self.get_pola(a, neu)
                    else:
                        pos_t = self.get_pola(a, neu)
                        neg_t = self.get_pola(a, pos)
                else:
                    ini_pola = ini_txt
                    pos_t = ini_txt
                    neg_t = ini_txt
                    neg1_t = ini_txt
                    print("not completed")
                tt = self.tokenizer.batch_encode_plus([ini_pola], max_length=self.hparams.max_seq_length,
                                                      pad_to_max_length=True, truncation=True, return_tensors="pt")
                pos_text = self.tokenizer.batch_encode_plus([pos_t], max_length=self.hparams.max_seq_length,
                                                            pad_to_max_length=True, truncation=True,
                                                            return_tensors="pt")
                neg_text = self.tokenizer.batch_encode_plus([neg_t], max_length=self.hparams.max_seq_length,
                                                            pad_to_max_length=True, truncation=True,
                                                            return_tensors="pt")

                input = tt['input_ids']
                mask = tt['attention_mask']
                pos_input = pos_text['input_ids']
                pos_ma = pos_text['attention_mask']
                neg_input = neg_text['input_ids']
                neg_ma = neg_text['attention_mask']
            else:
                tri = tar_txt.split('; ')
                j = 0
                for ti in tri:
                    if self.completed(ti, 2):
                        a, o, c = ti.split(', ')[:3]
                        ini_pola = a + '; ' + o
                        if c == 'positive':
                            pos_t = self.get_pola(a, pos)
                            neg_t = self.get_pola(a, neg)
                        elif c == 'negative':
                            pos_t = self.get_pola(a, neg)
                            neg_t = self.get_pola(a, neu)
                        else:
                            pos_t = self.get_pola(a, neu)
                            neg_t = self.get_pola(a, pos)
                    else:
                        ini_pola = ini_txt
                        pos_t = ini_txt
                        neg_t = ini_txt
                        print("not completed")
                    tt = self.tokenizer.batch_encode_plus([ini_pola], max_length=self.hparams.max_seq_length,
                                                          pad_to_max_length=True, truncation=True, return_tensors="pt")
                    pos_text = self.tokenizer.batch_encode_plus([pos_t], max_length=self.hparams.max_seq_length,
                                                                pad_to_max_length=True, truncation=True,
                                                                return_tensors="pt")
                    neg_text = self.tokenizer.batch_encode_plus([neg_t], max_length=self.hparams.max_seq_length,
                                                                pad_to_max_length=True, truncation=True,
                                                                return_tensors="pt")
                    if j == 0:
                        input = tt['input_ids']
                        mask = tt['attention_mask']
                        pos_input = pos_text['input_ids']
                        pos_ma = pos_text['attention_mask']
                        neg_input = neg_text['input_ids']
                        neg_ma = neg_text['attention_mask']
                    else:
                        input = torch.cat((input, tt['input_ids']), 0)
                        mask = torch.cat((mask, tt['attention_mask']), 0)
                        pos_input = torch.cat((pos_input, pos_text['input_ids']), 0)
                        pos_ma = torch.cat((pos_ma, pos_text['attention_mask']), 0)
                        neg_input = torch.cat((neg_input, neg_text['input_ids']), 0)
                        neg_ma = torch.cat((neg_ma, neg_text['attention_mask']), 0)
                    j += 1
            if i == 0:
                n_token = input
                n_mask = mask
                pos_token = pos_input
                pos_mask = pos_ma
                neg_token = neg_input
                neg_mask = neg_ma
            else:
                n_token = torch.cat((n_token, input), 0)
                n_mask = torch.cat((n_mask, mask), 0)
                pos_token = torch.cat((pos_token, pos_input), 0)
                pos_mask = torch.cat((pos_mask, pos_ma), 0)
                neg_token = torch.cat((neg_token, neg_input), 0)
                neg_mask = torch.cat((neg_mask, neg_ma), 0)

        return n_token, n_mask, pos_token, pos_mask, neg_token, neg_mask


    def sent(self, tokens):
        return tokens[:, 0]

    def replace_i(self, triple, idx, listt):
        randid = random.randint(0, len(listt)-1)
        triple[idx] = listt[randid]
        return ', '.join(triple)

    def replace_e(self, rel_idx, init_t, batch_t):
        rep_tri = ''
        if ';' not in batch_t:
            if self.completed(batch_t, rel_idx):
                it = init_t
                tri = batch_t.split(', ')
                listt = it.replace(tri[rel_idx], '').split(' ')
                rep_tri = self.replace_i(tri, rel_idx, listt)
            else:
                rep_tri = batch_t
                print("not completed")
        else:
            tri = batch_t.split('; ')
            t = []
            for tridx in tri:
                if self.completed(tridx, rel_idx):
                    it = init_t
                    triple = tridx.split(', ')
                    listt = it.replace(triple[rel_idx], '').split(' ')
                    x1 = self.replace_i(triple, rel_idx, listt)
                    t.append(x1)
                else:
                    t.append(tridx)
                    print("not completed")
            rep_tri = '; '.join(t)
        return rep_tri

    def completed(self, triple, idx):
        if len(triple.split(', ')) >= idx+1:
            return True

    def replace_s(self, rel_idx, batch_t, rep_set):
        rep_tri = ''
        if ';' not in batch_t:
            listt = rep_set.copy()
            if self.completed(batch_t, rel_idx):
                tri = batch_t.split(', ')
                if tri[rel_idx] in listt:
                    listt.remove(tri[rel_idx])
                rep_tri = self.replace_i(tri, rel_idx, listt)
            else:
                rep_tri = batch_t
                print("not completed")
        else:
            tri = batch_t.split('; ')
            t = []
            for tridx in tri:
                listt = rep_set.copy()
                if self.completed(tridx, rel_idx):
                    x = tridx.split(', ')
                    listt.remove(x[rel_idx])
                    xl = self.replace_i(x, rel_idx, listt)
                    t.append(xl)
                else:
                    t.append(tridx)
                    print("not completed")
            rep_tri = '; '.join(t)
        return rep_tri

    def replace_tri(self, n_batch, text, idx):
        randid = random.randint(0, n_batch - 1)
        if randid == idx:
            randid = random.randint(0, n_batch - 1)
        rep_tri = text[randid]
        return rep_tri

    def get_neg_token(self, token, ini, rep):
        texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in token]
        init = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in ini]
        n_batch = token.size(0)
        n_token = torch.zeros((1, self.hparams.max_seq_length), dtype=torch.long, device=f'cuda')
        n_mask = torch.zeros((1, self.hparams.max_seq_length), dtype=torch.long, device=f'cuda')

        for i in range(n_batch):
            batch_t = texts[i]
            ini_t = init[i]

            if rep == 'aspect':
                rep_tri = self.replace_e(rel_idx=0, init_t=ini_t, batch_t=batch_t)
            elif rep == 'opinion':
                rep_tri = self.replace_e(rel_idx=1, init_t=ini_t, batch_t=batch_t)
            elif rep == 'cate':
                rep_tri = self.replace_s(rel_idx=1, batch_t=batch_t, rep_set=aspect_cate_list)
            elif rep == 'pola':
                rep_tri = self.replace_s(rel_idx=2, batch_t=batch_t, rep_set=sentiment_word_list)
            elif rep == 'combine':
                batch_t = self.replace_e(rel_idx=0, init_t=ini_t, batch_t=batch_t)
                rep_tri = self.replace_e(rel_idx=1, init_t=ini_t, batch_t=batch_t)
            elif rep == 'combine_cate':
                batch_t = self.replace_e(rel_idx=0, init_t=ini_t, batch_t=batch_t)
                rep_tri = self.replace_s(rel_idx=1, batch_t=batch_t, rep_set=aspect_cate_list)
            else:
                rep_tri = self.replace_tri(n_batch=n_batch, text=texts, idx=i)

            txt_encoding = self.tokenizer(rep_tri, padding="max_length", max_length=self.hparams.max_seq_length,
                                          truncation=True, return_tensors="pt")

            txt_id = txt_encoding['input_ids']
            txt_mask = txt_encoding['attention_mask']
            if i == 0:
                n_token = txt_id
                n_mask = txt_mask
            else:
                n_token = torch.cat((n_token, txt_id), dim=0)
                n_mask = torch.cat((n_mask, txt_mask), dim=0)
        return n_token, n_mask

    def compute(self, pred_seqs, gold_seqs):
        """
        compute metrics for multiple tasks
        """
        assert len(pred_seqs) == len(gold_seqs)
        num_samples = len(gold_seqs)

        all_labels, all_predictions = [], []

        for i in range(num_samples):
            gold_list = extract_spans_extraction(gold_seqs[i])
            pred_list = extract_spans_extraction(pred_seqs[i])

            all_labels.append(gold_list)
            all_predictions.append(pred_list)

        print("\nResults of raw output")
        raw_scores = compute_f1_scores(all_predictions, all_labels)
        value = raw_scores['f1']
        return value

    def evl(self, batch):
        """
        Compute scores given the predictions and gold labels
        """
        outs = self.model.generate(input_ids=batch['source_ids'],
                                   attention_mask=batch['source_mask'],
                                   max_length=128
                                   )
        dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        targets = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]


        #labels_hat = torch.argmax(outs, dim=1)
        val_f1 = self.compute(dec, targets)

        return float(val_f1)

    def _step(self, batch, cl=False):
        lm_labels = batch["target_ids"].clone()
        de_inputs = batch["target_ids"].clone()
        # all labels set to -100 are ignored(masked)
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100
        loss = self(input_ids=batch["source_ids"],
                    attention_mask=batch["source_mask"],
                    labels=lm_labels,
                    decoder_input_ids=de_inputs,
                    decoder_attention_mask=batch["target_mask"],
                    cl=cl
                    )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.hparams.cl)
        #loss.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm_(self.project1.parameters(), max_norm=1.0)
        #tri_loss = self.dynamic_weight_adjustment(tri_loss)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        f1 = self.evl(batch)
        loss = self._step(batch)
        return {"val_loss": loss, "val_f1": torch.tensor(f1)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_f1 = torch.stack([x["val_f1"] for x in outputs]).mean()

        tensorboard_logs = {"val_loss": avg_loss, "val_f1": avg_f1}
        return {"avg_val_loss": avg_loss, "avg_val_f1": avg_f1, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if not self.trainer.use_tpu:
            # xm.optimizer_step(optimizer)
        # else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        gpus = self.hparams.n_gpu.split(',')
        if len(gpus) > 1:
            gpu_len = len(gpus)
        else:
            gpu_len = 1
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, gpu_len)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)