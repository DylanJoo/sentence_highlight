"""
Training functions for sentence highlighting, 
which includes two methods based on deep NLP pretrained models.

Methods:
    (1) Bert: Highlight prediction tasks.
        (*) Task1: sequence labeling
        (*) Task2: span detection
    (2) T5: highlight generation.
        (*) Task3: marks generation.

Packages requirments:
    - hugginface 
    - datasets 
"""
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    Trainer,
    HfArgumentParser,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding
)

from datasets import load_dataset, DatasetDict
from models import BertForHighlightPrediction
from trainers import BertTrainer, T5Trainer

# Arguments: (1) Model arguments (2) DataTraining arguments (3)
@dataclass
class OurModelArguments:

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default='bert-base-uncased')
    model_type: Optional[str] = field(default='bert-base-uncased')
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default='bert-base-uncased')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    # Cutomized arguments
    pooler_type: str = field(default="cls")
    temp: float = field(default=0.05)
    num_labels: float = field(default=2)


@dataclass
class OurDataArguments:

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    # Customized arguments
    train_file: Optional[str] = field(default="data/parsed/train/esnli_sents_highlight_contracdict.jsonl")
    eval_file: Optional[str] = field(default="data/parsed/dev/esnli_sents_highlight_contracdict.jsonl")
    test_file: Optional[str] = field(default="data/parsed/test/esnli_sents_highlight_contracdict.jsonl")
    max_seq_length: Optional[int] = field(default=512)

@dataclass
class OurTrainingArguments(TrainingArguments):
    output_dir: str = field(default='./models')
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    save_steps: int = field(default=1000)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=16)
    weight_decay: float = field(default=0.0)
    logging_dir: Optional[str] = field(default='./logs')
    warmup_steps: int = field(default=1000)



def main():
    """
    (1) Prepare parser with the 3 types of arguments
        * Detailed argument parser by kwargs
    (2) Load the corresponding tokenizer and config 
    (3) Load the self-defined models
    (4)
    """

    # Parseing argument for huggingface packages
    parser = HfArgumentParser((OurModelArguments, OurDataArguments, OurTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_datalcasses()

    # config and tokenizers
    # [TODO] If the additional arguments are fixed for putting in the function,
    # make it consistent to the function calls.
    config_kwargs = {
            "num_labels": model_args.num_labels,
            "output_hidden_states": True
    }
    tokenizer_kwargs = {
            # "cache_dir": model_args.cache_dir, 
            # "use_fast": model_args.use_fast_tokenizer
    }
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)


    # model 
    model_kwargs = {
            "cache_dir": model_args.cache_dir,
    }
    model = BertForHighlightPrediction.from_pretrained(
            model_args.model_name_or_path,
            config=config, 
            model_args=model_args
    )

    # Dataset 
    # (1) loading the jsonl file
    # (2) Preprocessing 
    # [CONCERN] Mode the prepare function outside this train.py 
    def preprare_esnli_seq_labeling(examples, aggregate_type='all'):
    
        def merge_list(word_labels, word_id_list):
            """aggregate the word_lables to token_labels."""
            token_labels = word_labels
            for i, idx in enumerate(word_id_list):
                if idx == None:
                    token_labels.insert(i, -100)
                elif word_id_list[i-1] == word_id_list[i]:
                    token_labels.insert(i, -100)

            return token_labels

        size = len(examples['sentA'])
        features = tokenizer(
            examples['sentA'], examples['sentB'],
            max_length=data_args.max_seq_length, # [TODO] make it callable
            truncation=True,
            padding=True,
        )   
        features['labels'] = [[0] * len(features['input_ids'][1])] * size

        for b in range(size):
            features['labels'][b] = merge_list(
                word_labels=examples['labels'][b], 
                word_id_list=features.word_ids(b)
            )

        return features

    dataset = DatasetDict.from_json({
        "train": data_args.train_file,
        "dev": data_args.eval_file
    })

    dataset = dataset.map(
        function=preprare_esnli_seq_labeling,
        batched=True,
        remove_columns=['sentA', 'sentB', 'keywordsA', 'keywordsB', 'labels'],
        num_proc=multiprocessing.cpu_count()
    )


    # Dataset
    # (2) data collator: transform the datset into the training mini-batch
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        return_tensors="pt",
        padding=True
    )

    # Trainer
    trainer = Trainer(
            model=model, 
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['dev'],
            data_collator=data_collator
    )
    # trainer = BertTrainer(
    #         model=model, 
    #         args=training_args,
    #         train_dataset=dataset['train'],
    #         eval_dataset=dataset['dev'],
    #         data_collator=data_collator
    # )
    trainer.model_args = model_args
    
    # ***** strat training *****
    results = trainer.train(model_path=model_args.model_name_or_path)



    return results

if __name__ == '__main__':
    main()
