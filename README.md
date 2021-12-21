# Sentence Highlight
Comparing deep contextualized model for sentences highlighting task. 
In addition, take the classic explanation model "LIME" with bert-base model as the baseline approach.

1. Term importance estimation
> Bert-LIME (See [OLD](https://github.com/DylanJoo/temp/tree/main/lime))
2. Learning to explain
> T5-marks-generation (TBA) \
> Bert-seq-labeling (THIS REPO)
<hr/>

**Repositary Updates**
- [2021/12/08]: See the quick experiments (prototype) on [colab notebook](https://colab.research.google.com/drive/14DxpHoSV7hL1YgrPPdVNIbp1aHeSKHgc?usp=sharing)
- [2021/12/15]: The latest evaluation results of bert-seq-labeling (10000 steps): 
> E-snli Dev sets \
> Mean precision: 0.8515797960428833            
> Mean recall   : 0.7201867890092417            
> Mean f1-score : 0.7329909700176722 

## Bert-LIME
<hr/>

## T5-marks-generation
<hr/>

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
- Inferencing
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
- Evaluation
```
# Evaluate bert-seq-labeling on esnli
python3 evaluation.py \
    -truth 'data/parsed/dev/esnli_sents_highlight_contradict.jsonl' \
    -pred 'results/esnli/bert-seq-labeling-dev.jsonl' \
    -hl_type 'bert-seq-labeling'

# Evalute t5-marks-generation
python3 evaluation.py \
    -truth 'data/parsed/dev/esnli_sents_highlight_contradict.jsonl' \
    -pred 'results/esnli/t5-marks-generation-dev.txt' \
    -hl_type 't5-marks-generation'
```
- Results 

*E-snli Dev set (#examples 3278)*
Methods | Mean Precision | Mean Precision | Mean F1-score
:------- |:--------------:|:--------------:|:-------------:
Bert-LIME           | -     | -     | -
Bert-seq-labeling   | 1.000 | 0.625 | 0.70
Bert-span-detection | 0.847 | 0.626 | 0.676

*E-snli Dev set (#examples 3237)*
Methods | Mean Precision | Mean Precision | Mean F1-score
:------- |:--------------:|:--------------:|:-------------:
Bert-LIME           | -     | -     | -
Bert-seq-labeling   | 0.000 | 0.000 | 0.000
Bert-span-detection | 0.856 | 0.644 | 0.691


```
Examples: \
 Ground truth tokens: ['men', 'fighting']              
 Highlighted tokens: ['men', 'fighting'] \
 Ground truth tokens: ['jackets', 'walk', 'to', 'school']               
 Highlighted tokens: ['jackets']
********************************            
Mean precision: 1.0              
Mean recall   : 0.625            
Mean f1-score : 0.7              
Num of evaluated samples: 2            
********************************
```

## T5-marks-generation
- Highlight dataset
- Training
- Evaluation
