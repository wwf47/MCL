CUDA_VISIBLE_DEVICSE=6 nohup python -u main.py \
          --task aste \
          --num_train_epochs 20 \
          --train_batch_size 32 \
          --gradient_accumulation_steps 1 \
          --model_name_or_path model/mt5-small \
          --learning_rate 1e-5 \
          --tcl_weight 0.4 \
          --scl_weight 0.4 \
          --k 1 \
          --do_train \
          --do_eval \
          --element mt5-small \
          --n_gpu 6 > out/run_mt5_aste.log 2>&1 &

