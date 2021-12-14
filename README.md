# Sentence Highlight
Comparing deep contextualized model for sentences highlighting task. 
In addition, take the classic explanation model "LIME" with bert-base model as the baseline approach.

1. Term importance estimation
> Bert-LIME ([OLD](https://github.com/DylanJoo/temp/tree/main/lime))
2. Learning to explain
> T5-marks-generation (TBA) \
> Bert-token-labeling (THIS REPO)
<hr/>

**Repositary Updates**
- [2021/12/08]: See the quick experiments (prototype) on [colab notebook](https://colab.research.google.com/drive/14DxpHoSV7hL1YgrPPdVNIbp1aHeSKHgc?usp=sharing)

## Bert-token-labeling
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
> Huggingface's Bert implementaion with PyTorch frameworks. \
```
python3 train.py \
  --model_name_or_path bert-base-uncased \
  --config_name bert-base-uncased \
  --output_dir ./models/bert-base-uncased \
  --max_steps 1000 \
   --train_file "data/parsed/train/esnli_sents_highligh_contradict.jsonl" \
  --eval_file "data/parsed/dev/esnli_sents_highligh_contradict.jsonl"
  --max_seq_length 128 \
  --do_train \
```
- Evaluation
```
python3 inference.py \
  --model_name_or_path "{Huggingface's CKPT}" \
  --config_name bert-base-uncased \
  --output_dir ./models/bert-base-uncased \
  --train_file "data/parsed/train/esnli_sents_highlight_contradict.jsonl" \
  --eval_file "data/parsed/dev/esnli_sents_highlight_contradict.jsonl" \
  --max_seq_length 128 \
  --do_eval 
```

## T5-marks-generation
- Highlight dataset
- Training
- Evaluation
