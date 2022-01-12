STEPS=10000
mkdir results/fin10k

for FILE_PATH in ./data/fin10k/*.jsonl
do
    echo "============================"
    echo "Start $FILE_PATH Highlighting"
    echo "============================"
    FILE=${FILE_PATH##*/}
    filecode=${FILE/.jsonl/""}

    python3 bert-seq-labeling/inference.py \
      --model_name_or_path checkpoints/bert-base-uncased/checkpoint-${STEPS} \
      --output_dir checkpoints/bert-base-uncased \
      --config_name bert-base-uncased \
      --eval_file $FILE_PATH \
      --result_json results/fin10k/bert-seq-labeling.${filecode}.highlights.jsonl \
      --max_seq_length 128 \
      --do_eval

    echo "============================"
    echo "Highlighting Completed"
    echo "RESULTS: results/fin10k/bert-seq-labeling.${filecode}.highlights.jsonl"
    echo "============================"
done

