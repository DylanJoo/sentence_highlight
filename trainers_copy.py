"""
Customized trainer for setnece highlight
"""
import multiprocessing
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

        # 1) Retrieve requirment of evaluation (words and word_ids)
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset # have already tokenized
        def merge_words(examples):
            features = {"words": [None] * len(examples['wordsA'])}
            for b in range(len(examples['wordsA'])):
                features['words'][b] = [None] + examples['wordsA'][b] + [None] + examples['wordsB'][b] + [None],
            return features

        words = eval_dataset.map(
            function=merge_words,
            batched=True,
            num_proc=multiprocessing.cpu_count()
        )['words']

        output_dict.update(dict(zip(range(len(words)), words)))

        eval_dataloader = self.get_eval_dataloader(eval_dataset.remove_columns(['wordsA', 'wordsB']))

        f = open(output_jsonl, 'w') 

        for b, batch in enumerate(eval_dataloader):
            for k in batch:
                batch[k] = batch[k].to(self.args.device)

            output = self.model.inference(batch)
            prob = (output['probabilities'] * output['active_tokens']).cpu().numpy()
            label = output['active_predictions'].cpu().numpy()

            for n in range(len(batch)):
                i_example = b * self.args.eval_batch_size + n
                predictions = collections.defaultdict(list)
                word_id_list = eval_dataset['word_ids'][i_example]

                for i, word_i in enumerate(word_id_list):
                    if word_i == None:
                        predictions['label'].append(-1)
                        predictions['prob'].append(-1)
                    elif word_id_list[i-1] == word_id_list[i]:
                        if prob_aggregate_strategy == 'max':
                            predictions['prob'][-1] = max(prob[n][i], predictions['prob'][-1])
                        if prob_aggregate_strategy == 'mean':
                            dist = (i - len(predictions['prob'])-1 )
                            predictions['prob'][-1] = \
                                    (predictions['prob'][-1] * dist + prob[n][i]) / (dist + 1)
                        else: 
                            pass 
                    else:
                        predictions['label'].append(label[n][i])
                        predictions['prob'].append(prob[n][i])

                output_dict[i_example].update(predictions)

                if save_to_json:
                    f.write(json.dumps(
                        predictions
                    ) + '\n')
            
            if b % 10 == 0:
                print(f"Evaluating {b} batches.")

        return output_dict




class T5Trainer(Trainer):

    def inference(self):
        pass

