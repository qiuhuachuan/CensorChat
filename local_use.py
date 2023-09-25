from typing import Optional

import torch
from transformers import BertConfig, BertTokenizer, BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

label_mapping = {0: 'NSFW', 1: 'SFW'}

config = BertConfig.from_pretrained('qiuhuachuan/CensorChat',
                                    num_labels=2,
                                    finetuning_task='text classification')
tokenizer = BertTokenizer.from_pretrained('qiuhuachuan/CensorChat',
                                          use_fast=False,
                                          never_split=['[user]', '[bot]'])
tokenizer.vocab['[user]'] = tokenizer.vocab.pop('[unused1]')
tokenizer.vocab['[bot]'] = tokenizer.vocab.pop('[unused2]')


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel.from_pretrained('bert-base-cased')
        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else
                              config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        # we use cls embedding
        cls = outputs[0][:, 0, :]
        cls = self.dropout(cls)
        logits = self.classifier(cls)

        loss = None

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model = BertForSequenceClassification.from_pretrained('qiuhuachuan/CensorChat',
                                                      config=config)
model.eval()

text = '''I'm open to exploring a variety of toys, including vibrators, wands, and clamps. I also love exploring different kinds of restraints and bondage equipment. I'm open to trying out different kinds of toys and exploring different levels of intensity.'''
result = tokenizer.encode_plus(text=text,
                               padding='max_length',
                               max_length=512,
                               truncation=True,
                               add_special_tokens=True,
                               return_token_type_ids=True,
                               return_tensors='pt')

with torch.no_grad():
    outputs = model(**result)
    predictions = outputs.logits.argmax(dim=-1)
    pred_label_idx = predictions.item()
    pred_label = label_mapping[pred_label_idx]
    print('predicted label is:', pred_label)