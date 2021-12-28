# bert-seq-labeling
python3 evaluation.py \
    -truth data/esnli.dev.sent_highlight.contradiction.jsonl \
    -pred results/esnli/bert-seq-labeling-dev.jsonl \
    -hl_type 'bert-seq-labeling'

python3 evaluation.py \
    -truth data/esnli.test.sent_highlight.contradiction.jsonl \
    -pred results/esnli/bert-seq-labeling-test.jsonl \
    -hl_type 'bert-seq-labeling'

# bert-lime
# python3 evaluation.py \
#     -truth data/esnli.dev.sent_highlight.contradiction.jsonl \
#     -pred results/esnli/bert-lime-top5-dev.jsonl \
#     -hl_type 'bert-lime'

# t5-marks-generation
# python3 evaluation.py \
#     -truth data/esnli.dev.sent_highlight.contradiction.jsonl \
#     -pred results/esnli/t5-marks-generation-dev.txt \
#     -hl_type 't5-marks-generation'

# t5-token-extractionn
# python3 evaluation.py \
#     -truth data/esnli.dev.sent_highlight.contradiction.jsonl \
#     -pred results/esnli/t5-marks-generation-dev.txt \
#     -hl_type 't5-token-extraction'
