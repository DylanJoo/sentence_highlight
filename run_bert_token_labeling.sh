# Finetunning for sentence highlight task with bert-token-labeling model 

python3 train.py \
  --model_name_or_path models/bert-for-token-labeling \
  --train_file data/parsed/train/esnli_sents_highlight_contradict.jsonl \
  --eval_file data/parsed/dev/esnli_sents_highlight_contradict.jsonl \
