import os

import numpy as np
import torch
import torch.nn as nn
from transformers import (BertPreTrainedModel,
                          DistilBertModel, DistilBertTokenizer, AutoTokenizer, AutoModel, BertModel,
                          BertForSequenceClassification, AdamW, BertModel, BertConfig)


def get_tokenizer_model(model_name: str = "google/bert_uncased_L-2_H-128_A-2",
                        config: BertConfig = None):
    model_name = download_once_pretrained_transformers(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return tokenizer, model


def save_pretrained_bert(model_name: str) -> str:

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model_path = f'./data/models/{model_name}/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    return model_path


def download_once_pretrained_transformers(
        model_name: str = "google/bert_uncased_L-2_H-128_A-2") -> str:
    model_path = f'./data/models/{model_name}/'

    if not os.path.exists(model_path):
        return save_pretrained_bert(model_name)

    return model_path


class BertForSequenceMultiLabelClassification(BertPreTrainedModel):
    """Constructed based on huggingface's BertForSequenceClassification 
    """

    def __init__(self, config):
        super(BertForSequenceMultiLabelClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                labels = torch.max(labels, 1)[1]
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
