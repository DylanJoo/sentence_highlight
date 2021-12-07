"""
Customized trainer for setnece highlight
"""
from transformers import Trainer


class BertTrainer(Trainer):

    def inference(self
                  eval_dataset, 
                  aggreagate_strategy):

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        # start evaluation (or testing)
        for i, batch in enumerate(eval_dataloader):
            output = self.model.inference(batch)

            # the selected (highlighted) probabilities.
            tokens_prob = output['probabilities'].cpu()
            tokens_pred = output['active_predictions'].cpu()
            tokens_mask = output['active_tokens'].cpu()

            # from token to words 
            for 



class BertTrainer(Trainer):

    def inference(self):
        pass
