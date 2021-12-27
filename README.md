# Sentence Highlight
Comparing deep contextualized model for sentences highlighting task. 
In addition, take the classic explanation model "LIME" with bert-base model as the baseline approach.

1. Term importance estimation
> Bert-LIME (See [OLD](https://github.com/DylanJoo/temp/tree/main/lime))
2. Learning to explain
> Bert-seq-labeling 
> T5-marks-generation (TBA) \
> T5-token-extraction (TBA) \
<hr/>

**Repositary Updates**
- [2021/12/08]: See the quick experiments (prototype) on [colab notebook](https://colab.research.google.com/drive/14DxpHoSV7hL1YgrPPdVNIbp1aHeSKHgc?usp=sharing)
- [2021/12/15]: The latest evaluation results of bert-seq-labeling (10000 steps): 

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
python3 bert-seq-labeling/train.py \
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
python3 bert-seq-labeling/inference.py \
  --model_name_or_path "{Huggingface's CKPT}" \
  --config_name bert-base-uncased \
  --output_dir ./models/bert-base-uncased \
  --eval_file "data/parsed/dev/esnli_sents_highlight_contradict.jsonl" \
  --test_file "data/parsed/test/esnli_sents_highlight_contradict.jsonl" \
  --max_seq_length 128 \
  --do_eval \
  --do_test 
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

**E-snli Dev set (#examples 3278)**
Methods  | Mean Precision | Mean Recall | Mean F1-score
:------- |:--------------:|:--------------:|:-------------:
Bert-LIME             | -     | -     | -
Bert-seq-labeling(.5) | 0.852 | 0.720 | 0.733
Bert-span-detection   | -     | -     | -
T5-marks-generation   | 0.847 | 0.626 | 0.676
T5-token-extraction   | 0.839 | 0.696 | 0.710

**E-snli Dev set (#examples 3237)**
Methods  | Mean Precision | Mean Recall | Mean F1-score
:------- |:--------------:|:--------------:|:-------------:
Bert-LIME             | -     | -     | -
Bert-seq-labeling(.5) | 0.853 | 0.734 | 0.744
Bert-span-detection   | -     | -     | -    
T5-marks-generation   | 0.856 | 0.644 | 0.691
T5-token-extraction   | 0.845 | 0.703 | 0.718

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
