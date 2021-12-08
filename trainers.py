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

        output_dict = collection.defaultdict(dict)

        # 1) Retrieve requirment of evaluation (words and word_ids)
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset # have already tokenized
        for i in range(len(eval_dataset)):
            output_dict[i] = {'words': [None] + eval_dataset['wordsA'] + [None] + eval_dataset['wordsB'] + [None]}

        eval_dataloader = self.get_eval_dataloader(eval_dataset.remove_columns(['wordsA', 'wordsB', 'word_ids']))

        f = open(output_jsonl, 'r') 

        for b, batch in enumerate(eval_dataloader):
            output = self.model.inference(batch)
            prob = (output['probabilities'] * output['active_tokens']).cpu().numpy()
            label = output['active_predictions'].cpu().numpy()

            for n in range(len(batch)):
                i_example = b * self.args.eval_batch_size + n
                predictions = collections.defaultdict(list)
                word_id_list = eval_dataset.word_ids(i_example)

                for i, word_i in enumerate(word_id_list):
                    if idx == None:
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
                    f.write(json.dumps(predictions) + '\n')

        return output_dict




class T5Trainer(Trainer):

    def inference(self):
        pass
