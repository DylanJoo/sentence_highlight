# example code, since this is my GCP's setting 
# You should change all the GCP related argument into yours

python3 t5/train.py \
  --base_dir gs://t541r/t5marks \
  --model_size 'base' \
  --task_type 'marks_generation' \
  --train_batch_size 16 \
  --train_steps 10000 \
  --max_src_len 512 \
  --max_tgt_len 64

# dev
python3 t5/inference.py \
  --input_file_gs gs://t541r/t5marks/data/dev/esnli_sents_highlight_contradict_pair.txt \
  --output_file 'results/esnli/t5-marks-generation-dev.txt' \
  --base_dir gs://t541r/t5marks \
  --task_type 'marks_generation' \
  --model_size 'base' 

# test
python3 t5/inference.py \
  --input_file_gs gs://t541r/t5marks/data/test/esnli_sents_highlight_contradict_pair.txt \
  --output_file 'results/esnli/t5-marks-generation-test.txt' \
  --base_dir gs://t541r/t5marks \
  --task_type 'marks_generation' \
  --model_size 'base' 
