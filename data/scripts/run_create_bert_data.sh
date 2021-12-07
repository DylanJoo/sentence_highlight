split=$1
python3 create_esnli_for_bert.py \
    -data_dir ../parsed/${split}/ \
    -output_path ../parsed/${split}/esnli_sents_highlight_all.jsonl \
    -highlightB_only \
    -class 'all'

python3 create_esnli_for_bert.py \
    -data_dir ../parsed/${split}/ \
    -output_path ../parsed/${split}/esnli_sents_highlight_contradict.jsonl \
    -highlightB_only \
    -class 'contradiction'
