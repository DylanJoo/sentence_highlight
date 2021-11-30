split=$1
python3 create_esnli_s2s_pair.py \
  -sentA ../parsed/${split}/sentenceA.txt \
  -sentB ../parsed/${split}/sentenceB.txt \
  -label ../parsed/${split}/label.txt \
  -highlightA ../parsed/${split}/highlightA.txt \
  -highlightB ../parsed/${split}/highlightB.txt \
  -out ../parsed/${split}/esnli_sents_highlight_contradict_pairs.tsv \
  -class 'contradiction' \
  -target highlight

python3 create_esnli_s2s_pair.py \
  -sentA ../parsed/${split}/sentenceA.txt \
  -sentB ../parsed/${split}/sentenceB.txt \
  -label ../parsed/${split}/label.txt \
  -highlightA ../parsed/${split}/highlightA.txt \
  -highlightB ../parsed/${split}/highlightB.txt \
  -out ../parsed/${split}/esnli_sents_highlight_all_pairs.tsv \
  -class 'all' \
  -target highlight

python3 create_esnli_s2s_pair.py \
  -sentA ../parsed/${split}/sentenceA.txt \
  -sentB ../parsed/${split}/sentenceB.txt \
  -label ../parsed/${split}/label.txt \
  -highlightA ../parsed/${split}/highlightA.txt \
  -highlightB ../parsed/${split}/highlightB.txt \
  -out ../parsed/${split}/esnli_sents_highlight_ctrl_pairs.tsv \
  -class 'all' \
  -target 'highlight_ctrl'
