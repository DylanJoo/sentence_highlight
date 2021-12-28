split=$1
python3 scripts/create_esnli_for_bert.py \
    -data_dir parsed \
    -split ${split} \
    -output_path esnli.${split}.sent_highlight.contradiction.jsonl \
    -highlightB_only \
    -class 'contradiction'

# The folowing codes will create the dataset on "ALL" classes (includes neutral and entailment)
# python3 scripts/create_esnli_for_bert.py \
#     -data_dir parsed \
#     -split ${split} \
#     -output_path esnli.${split}.sent_highlight.all.jsonl \
#     -highlightB_only \
#     -class 'all'

