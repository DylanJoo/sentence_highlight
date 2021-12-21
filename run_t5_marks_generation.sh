python3 t5-marks-generation/train.py \
  --base_dir gs://t541r/t5marks \
  --task_type 'marks-generation' \
  --model_size 'base' \
  --train_batch_size 16 \
  --train_steps 10000 \
  --max_src_len 512 \
  --max_tgt_len 64 

