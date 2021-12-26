# Finetunning for sentence highlight task with bert-token-labeling model 
# [TODO] Change the arugmnet of model name to self defined model, but the default tokenizer and configs
python3 bert-seq-labeling/train.py \
  --model_name_or_path checkpoints/bert-base-uncased \
  --output_dir ./models/bert-base-uncased/train \
  --config_name bert-base-uncased \
  --train_file data/parsed/train/esnli_sents_highlight_contradict.jsonl \
  --eval_file data/parsed/dev/esnli_sents_highlight_contradict.jsonl \
  --max_steps 10000 \
  --save_steps 1000 \
  --eval_steps 2000 \
  --evaluation_strategy 'steps'\
  --eval_steps 2000 \
  --max_seq_length 128 \
  --evaluate_during_training \
  --do_train \
  --do_eval

python3 bert-seq-labeling/inference.py \
  --model_name_or_path checkpoints/bert-base-uncased/checkpoint-${STEPS} \
  --config_name bert-base-uncased \
  --train_file data/parsed/train/esnli_sents_highlight_contradict.jsonl \
  --eval_file data/parsed/dev/esnli_sents_highlight_contradict.jsonl \
  --output_dir ./models/bert-base-uncased \
  --max_steps 10 \
  --max_seq_length 128 \
  --do_eval
