# Finetunning for sentence highlight task with bert-token-labeling model 
python3 bert-seq-labeling/train.py \
  --model_name_or_path 'bert-base-uncased' \
  --output_dir checkpoints/bert-base-uncased \
  --config_name 'bert-base-uncased' \
  --train_file data/esnli.train.sent_highlight.contradiction.jsonl \
  --eval_file data/esnli.dev.sent_highlight.contradiction.jsonl \
  --max_steps 10000 \
  --save_steps 1000 \
  --eval_steps 2000 \
  --evaluation_strategy 'steps'\
  --eval_steps 2000 \
  --max_seq_length 128 \
  --evaluate_during_training \
  --do_train \
  --do_eval

STEPS=10000
python3 bert-seq-labeling/inference.py \
  --model_name_or_path checkpoints/bert-base-uncased/checkpoint-${STEPS} \
  --output_dir checkpoints/bert-base-uncased \
  --config_name bert-base-uncased \
  --eval_file data/esnli.dev.sent_highlight.contradiction.jsonl \
  --test_file data/esnli.test.sent_highlight.contradiction.jsonl \
  --result_json results/esnli/bert-seq-labeling-split.jsonl \
  --max_steps 10 \
  --max_seq_length 128 \
  --do_eval \
  --do_test
