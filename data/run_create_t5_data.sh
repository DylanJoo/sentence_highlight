split=$1
python3 scripts/create_esnli_for_t5.py \
  -data_dir parsed \
  -split ${split} \
  -output_path esnli.${split}.sent_highlight.contradiction.tsv \
  -task 'marks-generation' \
  -class 'contradiction'

python3 scripts/create_esnli_for_t5.py \
  -data_dir parsed \
  -split ${split} \
  -output_path esnli.${split}.sent_keywords.contradiction.tsv \
  -task 'token-extraction' \
  -class 'contradiction'

# The following code is "conditional marks-generation" task
# python3 scripts/create_esnli_for_t5.py \
#   -data_dir parsed \
#   -split ${split} \
#   -output_path esnli.${split}.sent_highlight.conditional.tsv \
#   -task 'marks-generation-conditional' \
#   -class 'all' 

# The following code is "classification" task
# python3 scripts/create_esnli_for_t5.py \
#   -data_dir parsed \
#   -split ${split} \
#   -output_path esnli.${split}.sent_classficiation.tsv \
#   -target 'classification' \
#   -class 'all'

# The following code includes the classes "entailment and contradiction"
# python3 scripts/create_esnli_for_t5.py \
#   -data_dir parsed \
#   -split ${split} \
#   -output_path esnli.${split}.sent.highlight.contradiction_n_entailment.tsv \
#   -target 'marks-generation' \
#   -class 'contradiction or entailment' 

# The following code includes all the classes>
# python3 scripts/create_esnli_for_t5.py \
#   -data_dir parsed \
#   -split ${split} \
#   -output_path esnli.${split}.sent_highlight.all.tsv \
#   -target 'marks-generation' \
#   -class 'all' \

