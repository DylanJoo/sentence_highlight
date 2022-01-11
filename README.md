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
- [2021/12/08]: See the experiments on notebooks
> bert-seq-labeling: [colab notebook](https://colab.research.google.com/drive/14DxpHoSV7hL1YgrPPdVNIbp1aHeSKHgc?usp=sharing)
- [2021/12/17]: See the experiments on notebooks
> t5-marks-generation and t5-token-extraction: [colab notebook](https://colab.research.google.com/drive/1bQfOsrgu6lkgdro8SiLv2GD_hkeoHz0Q?usp=sharing)
- [2021/12/28]: Preprocessing pipeline updates and integrate the "infernecing" features on corpus from other domains.
- [2021/12/28]: Results updates as follow.

**E-snli Dev set (#examples 3278)**
Methods  | Mean Precision | Mean Recall | Mean F1-score
:------- |:--------------:|:--------------:|:-------------:
Bert-LIME             | -     | -     | -
Bert-seq-labeling(.5) | 0.852 | 0.720 | 0.733
Bert-span-detection   | -     | -     | -
T5-marks-generation   | 0.847 | 0.626 | 0.676
T5-token-extraction   | 0.839 | 0.696 | 0.710

**E-snli Test set (#examples 3237)**
Methods  | Mean Precision | Mean Recall | Mean F1-score
:------- |:--------------:|:--------------:|:-------------:
Bert-LIME             | -     | -     | -
Bert-seq-labeling(.5) | 0.853 | 0.734 | 0.744
Bert-span-detection   | -     | -     | -    
T5-marks-generation   | 0.856 | 0.644 | 0.691
T5-token-extraction   | 0.845 | 0.703 | 0.718

<hr/>

## Unsupervised Learning - Bert-LIME Esimtation
- TBA

<hr/>

## Supervised Learning
- E-snli dataset file downlaoding and parsing
> Download the files from 'OanaMariaCamburu/e-SNLI'
```
cd data
bash scripts/download_esnli.sh
bash run_parse.sh
```
- Preparing E-snli dataset into the corresponding models and tasks
> (1) bert-seq-labeling

```
cd data
bash run_create_bert_data.sh 'train'
bash run_create_bert_data.sh 'dev'
bash run_create_bert_data.sh 'test'
```
> (2) t5-marks-generation
> (3) t5-token-extraction

```
cd data
bash run_create_t5_data.sh 'train'
bash run_create_t5_data.sh 'dev'
bash run_create_t5_data.sh 'test'
```

### Bert-seq-labeling
- Training
> Huggingface's Bert implementaion with PyTorch frameworks. Besides, I recommend using the [Weights & Biases](https://wandb.ai/) as the visualization tools, and connect them into the training process!
```
python3 bert-seq-labeling/train.py \
  --model_name_or_path 'bert-base-uncased' \
  --output_dir checkpoints/bert-base-uncased \
  --config_name 'bert-base-uncased' \
  --train_file data/esnli.train.sent_highlight.contradiction.jsonl \
  --eval_file data/esnli.dev.sent_highlight.contradiction.jsonl \
  --max_steps 10000 \
  --save_steps 1000 \
  --eval_steps 2000 \
  --evaluation_strategy 'steps'\
  --eval_steps 2000 \
  --max_seq_length 128 \
  --evaluate_during_training \
  --do_train \
  --do_eval
```
> You now can find out the checkpoint in the 'checkpoints' folder or you can download our [finetuned checkpoints](#)

- Inferencing
```
STEPS=10000
python3 bert-seq-labeling/inference.py \
  --model_name_or_path checkpoints/bert-base-uncased/checkpoint-${STEPS} \
  --output_dir checkpoints/bert-base-uncased \
  --config_name bert-base-uncased \
  --eval_file data/esnli.dev.sent_highlight.contradiction.jsonl \
  --test_file data/esnli.test.sent_highlight.contradiction.jsonl \
  --result_json results/bert-seq-labeling-split.jsonl \
  --max_steps 10 \
  --max_seq_length 128 \
  --do_eval \
  --do_test
```
> You now can find out the esnli dev/test results in results/esnli.

- Evaluation 
```
# dev set
python3 evaluation.py \
    -truth 'data/esnli.dev.sent_highlight_contradiction.jsonl' \
    -pred 'results/esnli/bert-seq-labeling-dev.jsonl' \
    -hl_type 'bert-seq-labeling'

# test set 
python3 evaluation.py \
    -truth 'data/esnli.dev.sent_highlight_contradiction.jsonl' \
    -pred 'results/esnli/bert-seq-labeling-dev.jsonl' \
    -hl_type 'bert-seq-labeling'
```

### T5-marks-generation
### T5-token-extraction

- Evaluation
```
# Evalute t5-marks-generation
python3 evaluation.py \
    -truth 'data/parsed/dev/esnli_sents_highlight_contradict.jsonl' \
    -pred 'results/esnli/t5-marks-generation-dev.txt' \
    -hl_type 't5-marks-generation'
```

## T5-marks-generation
- Highlight dataset
- Training
- Evaluation
