split=$1
# python3 create_esnli_for_t5.py \
#   -data_dir ../parsed/${split} \
#   -output_path ../parsed/${split}/esnli_sents_classficiation.tsv \
#   -target 'classification' \
#   -class 'all'

python3 create_esnli_for_t5.py \
  -data_dir ../parsed/${split} \
  -output_path ../parsed/${split}/esnli_sents_highlight_contradict_pairs.tsv \
  -target 'highlight' \
  -class 'contradiction'

python3 create_esnli_for_t5.py \
  -data_dir ../parsed/${split} \
  -output_path ../parsed/${split}/esnli_sents_highlight_contradict_n_entailment_pairs.tsv \
  -target 'highlight' \
  -class 'contradiction or entailment' 

python3 create_esnli_for_t5.py \
  -data_dir ../parsed/${split} \
  -output_path ../parsed/${split}/esnli_sents_highlight_contradict_extraction_pairs.tsv \
  -target 'highlight_extraction' \
  -class 'contradiction'

python3 create_esnli_for_t5.py \
  -data_dir ../parsed/${split} \
  -output_path ../parsed/${split}/esnli_sents_highlight_conditional_pairs.tsv \
  -target 'highlight_conditional' \
  -class 'all' 

# python3 create_esnli_for_t5.py \
#   -data_dir ../parsed/${split} \
#   -output_path ../parsed/${split}/esnli_sents_highlight_all_pairs.tsv \
#   -target 'highlight' \
#   -class 'all' \

