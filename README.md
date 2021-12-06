# Sentence Highlight
Comparing deep contextualized model for sentences highlighting task. 
In addition, take the classic explanation model "LIME" with bert-base model as the baseline approach.

## Term importance estimation
    * Bert-LIME

## Proposed Methods (learning to explain)
    * Bert-seq-labeling
    * T5-marks-generation
## Bert-seq-labeling
#### Pipeline
- Highlight dataset
```
# Download the files from 'OanaMariaCamburu/e-SNLI'
bash download_esnli.sh
# Parsing the csv file into text file
bash run_parse.sh
# Preprocessing into the bert-token classfication task.
bash run_create_highlight_list.sh
```
- Training
```
python3 train.py \
  --train_file "data/parsed/train/esnli_sents_highligh_contradict.jsonl" \
  --eval_file "data/parsed/dev/esnli_sents_highligh_contradict.jsonl"
```
- Evaluation
