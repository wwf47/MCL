CUDA_VISIBLE_DEVICSE=2 nohup python -u main.py \
          --task acsd \
          --num_train_epochs 30 \
          --train_batch_size 16 \
          --gradient_accumulation_steps 2 \
          --model_name_or_path model/t5-base \
          --k 1 \
          --tcl_weight 0.4 \
          --scl_weight 0.5 \
          --do_train \
          --do_eval \
          --element ini \
          --n_gpu 2 > out/run_acsd.log 2>&1 &

