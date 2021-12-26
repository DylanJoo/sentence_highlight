python3 evaluation.py \
    -truth data/parsed/dev/esnli_sents_highlight_contradict.jsonl \
    -pred results/esnli/bert-lime-top5-dev.jsonl \
    -hl_type 'bert-lime' 

python3 evaluation.py \
    -truth data/parsed/dev/esnli_sents_highlight_contradict.jsonl \
    -pred results/esnli/bert-seq-labeling-dev.jsonl \
    -hl_type 'bert-seq-labeling' 

python3 evaluation.py \
    -truth data/parsed/dev/esnli_sents_highlight_contradict.jsonl \
    -pred results/esnli/t5-marks-generation-dev.txt \
    -hl_type 't5-marks-generation' 

python3 evaluation.py \
    -truth data/parsed/dev/esnli_sents_highlight_contradict.jsonl \
    -pred results/esnli/t5-marks-generation-dev.txt \
    -hl_type 't5-token-extraction' 
