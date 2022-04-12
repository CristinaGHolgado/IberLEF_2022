# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:43:43 2022

@author: Cristina GH
"""


import torch
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModel



class BertClassifier(nn.Module):

    def __init__(self, lm, dropout=0.5):
        '''
        lm : STRING
            Langauage model name from hugging face
        '''

        super(BertClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(lm)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        with torch.no_grad():
          _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer