import argparse
import os
import logging
import pickle
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers import T5Tokenizer, T5Tokenizer, MBartTokenizer, BartTokenizer

from model import T5FineTuner

from data_utils import ABSADataset
from data_utils import write_results_to_log, read_line_examples_from_file
from eval_utils import compute_scores
logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)
def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='aste', type=str, 
                        help="The name of the task, selected from: [aste, acsd]")
    '''parser.add_argument("--dataset", default='rest14', type=str, 
                        help="The name of the dataset, selected from: [laptop14, rest14, rest15, rest16]")'''
    parser.add_argument("--model_name_or_path", default='model/t5-base', type=str,
                        help="Path to pre-trained model or shortcut name, selected from: [t5-base, bart-large, mt5-base]")
    parser.add_argument("--out", default='./outputs', type=str,
                        help="Path to out model")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev/test set.")

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # CL parameters
    parser.add_argument('--k', type=int, default=10, help="numbers of generate negative example")
    parser.add_argument('--T', type=float, default=0.07, help="tempreture")
    parser.add_argument('--cl', action='store_true', help='do contrastive learning')
    parser.add_argument('--tcl', type=bool, default=False, help="whether or not use constractive learning on triplet-view")
    parser.add_argument('--scl', type=bool, default=False, help="whether or not use constractive learning on sentiment-view")
    parser.add_argument('--start_epoch', type=int, default=0, help="which step start to use CL")
    parser.add_argument("--element", default='aspect', type=str,
                        help="The name of the elements in triple, selected from: [aspect, opinion, cate, pola, tri]")
    parser.add_argument("--pool_type", default='avg', type=str,
                        help="pooling type, selected from: [cls, cls_before_pooler, avg, avg_top2, avg_first_last]")
    parser.add_argument('--ema_decay', type=float, default=0.75, help="decay of ema")
    parser.add_argument("--tcl_weight", default=0.4, type=float)
    parser.add_argument("--scl_weight", default=0.4, type=float)
    parser.add_argument("--o_weight", default=0.1, type=float)
    parser.add_argument("--alpha", default=0.2, type=float)

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--dropt", default=0.1, type=float)
    parser.add_argument("--gen", default=False, type=bool)



    args = parser.parse_args()

    # set up output dir which looks like './aste/rest14/extraction/'
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    args.device = torch.device("cuda:{}".format(args.n_gpu))

    return args

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))

def evaluate(data_loader, model, tokenizer, sents, task, device):
    """
    Compute scores given the predictions and gold labels
    """
    #device = torch.device(f'cuda:{args.n_gpu}')
    #model.model.to(device)
    model.model.to(device)
    
    model.model.eval()
    outputs, targets = [], []
    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = model.model.generate(input_ids=batch['source_ids'].to(device),
                                    attention_mask=batch['source_mask'].to(device),
                                    max_length=128)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)

    #raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(outputs, targets, sents, task)
    raw_scores, fixed_scores, all_labels, all_preds, all_preds_fixed = compute_scores(outputs, targets, sents, task)
    # results = {'raw_scores': raw_scores, 'labels': all_labels, 'preds': all_preds}
    results = {'raw_scores': raw_scores, 'fixed_scores': fixed_scores, 'labels': all_labels,
               'preds': all_preds, 'preds_fixed': all_preds_fixed}
    pickle.dump(results, open(f"{args.output_dir}/results-{args.task}-{args.dataset}.pickle", 'wb'))

    return raw_scores, fixed_scores

def get_dataset(task):
    datasets = []
    if task == "aste":
        datasets = ["rest14", "laptop14"]
    elif task == "acsd":
        datasets = ["rest15", "rest16"]
    else:
        pass

    return  datasets

def main(args):
    if args.do_train:
        print("\n****** Conduct Training ******")
        model = T5FineTuner(args)
        '''model.prepare()
        non_optimizer_list = [model.target_encoder, model.target_pooler]
        for layer in non_optimizer_list:
            for para in layer.parameters():
                para.requires_grad = False'''
        if args.task == "aste":
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min', save_top_k=5)
        else:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min', save_top_k=5)

    # prepare for trainer
        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            gradient_clip_val=1.0,
            # amp_level='O1',
            max_epochs=args.num_train_epochs,
            checkpoint_callback=checkpoint_callback,
            callbacks=[LoggingCallback()],
            # distributed_backend='ddp'
        )
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # save the final model
        model.model.save_pretrained(args.output_dir)

        print("Finish training and saving the model!")
    
    if args.do_eval:
        print("\n****** Conduct Evaluating ******")

        # model = T5FineTuner(args)
        dev_results, test_results = {}, {}
        best_f1, best_checkpoint, best_epoch = -999999.0, None, None
        all_checkpoints, all_epochs = [], []

        # retrieve all the saved checkpoints for model selection
        saved_model_dir = args.output_dir
        for f in os.listdir(saved_model_dir):
            file_name = os.path.join(saved_model_dir, f)
            if 'cktepoch' in file_name:
                all_checkpoints.append(file_name)

        # conduct some selection (or not)
        print(f"We will perform validation on the following checkpoints: {all_checkpoints}")

        # load dev and test datasets
        dev_dataset = ABSADataset(args.tokenizer, data_dir=args.dataset, data_type='dev',
                        task=args.task, max_len=args.max_seq_length)
        dev_loader = DataLoader(dev_dataset, batch_size=32, num_workers=4)

        test_dataset = ABSADataset(args.tokenizer, data_dir=args.dataset, data_type='test', 
                        task=args.task, max_len=args.max_seq_length)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
        
        for checkpoint in all_checkpoints:
            epoch = checkpoint.split('=')[-1][:-5] if len(checkpoint) > 1 else ""
            # only perform evaluation at the specific epochs ("15-19")
            # eval_begin, eval_end = args.eval_begin_end.split('-')
            print(epoch)
            if 0 <= float(epoch) < 100:
                all_epochs.append(epoch)

                # reload the model and conduct inference
                print(f"\nLoad the trained model from {checkpoint}...")
                model_ckpt = torch.load(checkpoint)
                model = T5FineTuner(model_ckpt['hyper_parameters'])
                model.load_state_dict(model_ckpt['state_dict'])
                sents, _, _ = read_line_examples_from_file(f'data/{args.task}/{args.dataset}/dev.txt')
                dev_result, _ = evaluate(dev_loader, model, args.tokenizer, sents, args.task, args.device)
                if dev_result['f1'] > best_f1:
                    best_f1 = dev_result['f1']
                    best_checkpoint = checkpoint
                    best_epoch = epoch

                # add the global step to the name of these metrics for recording
                # 'f1' --> 'f1_1000'
                dev_result = dict((k + '_{}'.format(epoch), v) for k, v in dev_result.items())
                dev_results.update(dev_result)
                sents, _, _ = read_line_examples_from_file(f'data/{args.task}/{args.dataset}/test.txt')
                test_result, _ = evaluate(test_loader, model, args.tokenizer, sents, args.task, args.device)
                test_result = dict((k + '_{}'.format(epoch), v) for k, v in test_result.items())
                test_results.update(test_result)

        # print test results over last few steps
        print(f"\n\nThe best checkpoint is {best_checkpoint}")
        best_step_metric = f"f1_{best_epoch}"
        print(f"F1 scores on test set: {test_results[best_step_metric]:.4f}")

        print("\n* Results *:  Dev  /  Test  \n")
        metric_names = ['f1', 'precision', 'recall']
        for epoch in all_epochs:
            print(f"Epoch-{epoch}:")
            for name in metric_names:
                name_step = f'{name}_{epoch}'
                print(f"{name:<10}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}", sep='  ')
            print()

        results_log_dir = './results_log'
        if not os.path.exists(results_log_dir):
            os.mkdir(results_log_dir)
        log_file_path = f"{results_log_dir}/{args.task}-{args.dataset}.txt"
        write_results_to_log(log_file_path, test_results[best_step_metric], args, dev_results, test_results, all_epochs)


if __name__ == '__main__':
    # initialization
    args = init_args()
    seed_everything(args.seed)
    if "mt5" in args.model_name_or_path:
        args.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    elif "t5" in args.model_name_or_path:
        args.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    elif "mbart" in args.model_name_or_path:
        args.tokenizer = MBartTokenizer.from_pretrained(args.model_name_or_path)
    elif "bart" in args.model_name_or_path:
        args.tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    else:
        print("there are something wrong with tokenizer")
    # show one sample to check the sanity of the code and the expected output
    # training process
    datasets = get_dataset(args.task)
    for data in datasets:
        args.dataset = data
        print("\n", "=" * 30, f"NEW EXP: {args.task.upper()} on {args.dataset}", "=" * 30)
        print("\n", "=" * 30,
              f"cl is {args.cl} k: {args.k} epochs: {args.num_train_epochs} batch_size: {args.train_batch_size*args.gradient_accumulation_steps}",
              "=" * 30)
        print(f"Here is an example (from dev set) :")
        dataset = ABSADataset(tokenizer=args.tokenizer, data_dir=args.dataset, data_type='dev',
                              task=args.task, max_len=args.max_seq_length)
        data_sample = dataset[0]  # a random data sample
        s = args.tokenizer.encode
        print('Input :', args.tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
        print('Output:', args.tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))
        element_dir = os.path.join(args.out, args.element)
        if not os.path.exists(element_dir):
            os.mkdir(element_dir)
        task_dir = os.path.join(element_dir, args.task)

        if not os.path.exists(task_dir):
            os.mkdir(task_dir)

        task_dataset_dir = os.path.join(task_dir, args.dataset)
        if not os.path.exists(task_dataset_dir):
            os.mkdir(task_dataset_dir)
        args.output_dir = task_dataset_dir
        main(args)