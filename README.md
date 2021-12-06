# Sentence Highlight
Comparing deep contextualized model for sentences highlighting task. 
In addition, take the classic explanation model "LIME" with bert-base model as the baseline approach.

1. Term importance estimation
> Bert-LIME
2. Learning to explain
> T5-marks-generation\
> Bert-seq-labeling

## Bert-seq-labeling
- Highlight dataset
> a) Download the files from 'OanaMariaCamburu/e-SNLI'\
> b) Parsing the csv file into text file\
> c) Preprocessing into the bert-token classfication task.
```
bash download_esnli.sh
bash run_parse.sh
bash run_create_highlight_list.sh
```
- Training
> Huggingface's Bert implementaion with PyTorch frameworks.
```
python3 train.py \
  --train_file "data/parsed/train/esnli_sents_highligh_contradict.jsonl" \
  --eval_file "data/parsed/dev/esnli_sents_highligh_contradict.jsonl"
```
- Evaluation
