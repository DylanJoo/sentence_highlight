from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import torch


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

    size = len(examples['wordsA'])
    features = tokenizer(
      examples['wordsA'], examples['wordsB'],
      is_split_into_words=True, # allowed the pre-tokenization process, to match the seq-order
      max_length=64,
      truncation=True,
      padding=True,
    )
    features['labels'] = [None] * size
    features['word_ids'] = [None] * size

    for b in range(size):
      features['labels'][b] = merge_list(
          word_labels=examples['labels'][b],
          word_id_list=features.word_ids(b)
      )
      features['word_ids'][b] = features.word_ids(b)
    return features
  

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = DatasetDict.from_json({'train': "data/parsed/train/test.jsonl", 'dev': 'data/parsed/train/test.jsonl'})
dataset['train'] = dataset['train'].map(
        function=preprare_esnli_seq_labeling,
        batched=True,
        remove_columns=['sentB']
)
dataset['dev'] = dataset['dev'].map(
        function=preprare_esnli_seq_labeling,
        batched=True,
        remove_columns=['sentA']
)
print(dataset)
# inputs = {'words':{}, 'word_ids': {}}
# inputs['words'] = {
#         i: dataset['word_ids'][i] for i in range(len(dataset))
# }
# inputs['word_ids'] = {
#         i: [None] + dataset['wordsA'][i] + [None] + dataset['wordsB'][i] for i in range(len(dataset))
# }
# print(list(inputs['words'].values())[:5])
# print(list(inputs['word_ids'].values())[:5])
# # dataset.set_format(type='torch')
# # print("--")
# # print(dataset_wordsA[:5])
# # print(dataset_wordsB[:5])
# # print("--")
# # print(dataset['input_ids'][:5])
# # print("--")
# # print(isinstance(dataset, torch.utils.data.IterableDataset))
