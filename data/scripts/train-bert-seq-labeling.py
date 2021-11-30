from transformers import AutoTokenizer
from datasets import Dataset


def main():
    # prepare function
    def preprare_esnli_seq_labeling(examples, aggregate_type='all'):
        size = len(examples['sentA'])

        features = tokenizer(
            examples['sentA'], examples['sentB'],
            max_length=512, # [TODO] make it callable
            truncation=True,
            padding=True,
        )   

        def merge_list(word_labels, word_id_list):
            token_labels = word_labels
            for i, idx in enumerate(word_id_list):
                if idx == None:
                    token_labels.insert(i, -100)
                elif word_id_list[i-1] == word_id_list[i]:
                    token_labels.insert(i, -100)
            return token_labels

        features['token_labels'] = [[0] * len(features['input_ids'][1])] * size

        for b in range(size):
            features['token_labels'][b] = merge_list(
                word_labels=examples['labels'][b], 
                word_id_list=features.word_ids(b)
            )

        return features

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # dataset
    dataset = Dataset.from_json(
        "preprocessed/train/esnli_sents_highlight_contradict.jsonl",
        split='train',
    )
    dataset = dataset.add_column("token_labels", [[]] * len(dataset))
    dataset = dataset.map(
        function=preprare_esnli_seq_labeling,
        batched=True,
        remove_columns=['sentA', 'sentB', 'keywordsA', 'keywordsB', 'labels']
    )
    dataset = dataset.rename_columns({"token_labels": "labels"})
    print(dataset)

    # data collator


main()
