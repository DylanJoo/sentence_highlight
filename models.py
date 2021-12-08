"""
(1) BertForHighlightPrediction
(2) BertForHighlightSpanDetection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertModel, BertPreTrainedModel

class BertForHighlightPrediction(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.tokens_clf = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        tokens_output = outputs[0]
        highlight_logits = self.tokens_clf(self.dropout(tokens_output))
        highlight_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = highlight_logits.view(-1, self.num_labels)
            active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            highlight_loss = loss_fct(active_logits, active_labels)

        return TokenClassifierOutput(
            loss=highlight_loss,
            logits=highlight_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def inference(self, inputs):

        with torch.no_grad():
            outputs = self.forward(**inputs)
            probabilities = self.softmax(self.tokens_clf(outputs.hidden_states[-1]))
            predictions = torch.argmax(probabilities, dim=-1)

            # active filtering
            active_tokens = inputs['attention_mask'] == 0
            active_predictions = torch.where(
                active_tokens,
                predictions,
                torch.tensor(-1).type_as(predictions)
            )
            return {"probabilities": probabilities[:, :, 1].detach().numpy(), # shape: (batch, length)
                    "active_predictions": predictions.detach().numpy(),
                    "active_tokens": active_tokens}

