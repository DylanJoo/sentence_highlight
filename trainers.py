"""
Customized trainer for setnece highlight
"""
from transformers import Trainer
import json

class BertTrainer(Trainer):

    def inference(self,
                  output_jsonl='results.jsonl'
                  eval_dataset=None, 
                  prob_aggregate_strategy='first',
                  save_to_json=True):

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        output_predictions = dict()
        f = open(output_jsonl, 'r') 

        for i, batch in enumerate(eval_dataloader):
            output = self.model.inference(batch)

            # the selected (highlighted) probabilities.
            prob = (output['probabilities'] * output['active_tokens']).cpu().numpy()
            label = output['active_predictions'].cpu().numpy()
            # tokens_mask = output['active_tokens'].cpu()

            # from token to words 
            for b in range(len(batch)):
                word_id_list = batch.word_ids(b)
                token_list = batch.tokens(b)
                predictions = collections.defaultdict(list)
                predictions['word'] = [None] + batch['wordA'] + [None] + batch['wordB'] + [None]
                for i, (idx, token) in enumerate(word_id_list, token_list): 
                    if idx == None:
                        predictions['label'].append(-1)
                        predictions['prob'].append(-1)
                    elif word_id_list[i-1] == word_id_list[i]:
                        if prob_aggregate_strategy == 'max':
                            predictions['prob'][idx] = max(prob[b][i], predictions['prob'][idx])
                        if prob_aggregate_strategy == 'mean':
                            dist = (idx - i)
                            predictions['prob'][idx] = \
                                    (predictions['prob'][idx] * dist + prob[b][i]) / (dist + 1)
                        else: 
                            pass 
                    else:
                        predictions['label'].append(label[b][i])
                        predictions['prob'].append(prob[b][i])

                output_predictions[i] = predictions

                if save_to_json:
                    f.write(json.dumps(predictions) + '\n')


      return output_predictions




class BertTrainer(Trainer):

    def inference(self):
        pass
