split=$1
python3 create_esnli_token_clf.py \
    -sentA ../parsed/${split}/sentenceA.txt \
    -sentB ../parsed/${split}/sentenceB.txt \
    -highlightA ../parsed/${split}/highlightA.txt \
    -highlightB ../parsed/${split}/highlightB.txt \
    -label ../parsed/${split}/label.txt \
    -out ../parsed/${split}/esnli_sents_highlight_all.jsonl \
    -class 'all'

python3 create_esnli_token_clf.py \
    -sentA ../parsed/${split}/sentenceA.txt \
    -sentB ../parsed/${split}/sentenceB.txt \
    -highlightA ../parsed/${split}/highlightA.txt \
    -highlightB ../parsed/${split}/highlightB.txt \
    -label ../parsed/${split}/label.txt \
    -out ../parsed/${split}/esnli_sents_highlight_contradict.jsonl \
    -class 'contradiction'
