# Finetunning for sentence highlight task with bert-token-labeling model 
# [TODO] Change the arugmnet of model name to self defined model, but the default tokenizer and configs
# python3 train.py \
#   --model_name_or_path bert-base-uncased \
#   --config_name bert-base-uncased \
#   --train_file data/parsed/train/esnli_sents_highlight_contradict.jsonl \
#   --eval_file data/parsed/dev/esnli_sents_highlight_contradict.jsonl \
#   --output_dir ./models/bert-base-uncased \
#   --max_steps 1000 \
#   --max_seq_length 128 \
#   --do_train \
#   --do_eval

  # --model_name_or_path models/bert-for-token-labeling \

python3 evaluate.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --train_file data/parsed/train/esnli_sents_highlight_contradict.jsonl \
  --eval_file data/parsed/dev/esnli_sents_highlight_contradict.jsonl \
  --output_dir ./models/bert-base-uncased \
  --max_steps 10 \
  --max_seq_length 128 \
  --do_eval
