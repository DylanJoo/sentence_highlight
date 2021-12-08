# Finetunning for sentence highlight task with bert-token-labeling model 

python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --train_file data/parsed/train/esnli_sents_highlight_contradict.jsonl \
  --eval_file data/parsed/dev/esnli_sents_highlight_contradict.jsonl \
  --max_seq_length 128 \
  --num_train_epochs 1  \
  --do_train \
  --do_eval

  # --model_name_or_path models/bert-for-token-labeling \
