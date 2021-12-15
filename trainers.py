"""
Customized trainer for setnece highlight
"""
from transformers import Trainer
import time
import json
import collections

class BertTrainer(Trainer):

    def inference(self,
                  output_jsonl='results.jsonl',
                  eval_dataset=None, 
                  prob_aggregate_strategy='first',
                  save_to_json=True):

        output_dict = collections.defaultdict(dict)
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        def merge_words(examples):
            features = {'words': [None] * len(examples['wordsA']) }
            for b in range(len(examples['wordsA'])):
                features['words'][b] = \
                        ["<tag1>"] + examples['wordsA'][b] + \
                        ["<tag2>"] + examples['wordsB'][b] + ["<tag3>"]
            return features

        # the tokenized words (to-be-scored) 
        # [CONCERN] Process the token in the beginning ?
        words = eval_dataset.map(
                function=merge_words, 
                batched=True,
                num_proc=multiprocessing.cpu_count()
        )['words']

        f = open(output_jsonl, 'r')
        eval_dataloader = self.get_eval_dataloader(
                eval_dataset.remove_columns(['wordsA', 'wordsB'])
        )
        for b, batch in enumerate(eval_dataloader):
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            output = self.model.inference(batch)
            prob = (output['probabilities'] * output['active_tokens']).cpu().tolist()
            label = output['active_predictions'].cpu().tolist()

            # per example in batch
            for n in range(self.args.eval_batch_size):
                i_example = b * self.args.eval_batch_size + n
                predictions = collections.defaultdict(list)
                predictions['word'] += words[i_example]
                word_id_list = eval_dataset['word_ids'][i_example]

                for i, word_i in enumerate(word_id_list):
                    if idx == None:
                        predictions['label'].append(-1)
                        predictions['prob'].append(-1)

                    elif word_id_list[i-1] == word_id_list[i]:
                        if prob_aggregate_strategy == 'max':
                            predictions['prob'][-1] = max(prob[n][i], predictions['prob'][-1])
                        if prob_aggregate_strategy == 'mean':
                            dist = (i - len(predictions['prob']) - 1)
                            predictions['prob'][-1] = \
                                    (predictions['prob'][-1] * dist + prob[n][i]) / (dist + 1)
                        else: 
                            pass 
                    else:
                        predictions['label'].append(label[n][i])
                        predictions['prob'].append(prob[n][i])

                output_dict[i_example] = predictions

                if save_to_json:
                    f.write(json.dumps(predictions) + '\n')

            if b % 100 = 0:
                print(f"Inferencing batch: {b}")
                print(f"words: {predictions['word']}")
                print(f"labels: {predictions['label']")
                print(f"probs: {predictions['prob']}")

        return output_dict




class T5Trainer(Trainer):

    def inference(self):
        pass
