# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:43:43 2022

@author: Cristina GH
"""


import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel



class BertClassifier(nn.Module):

    def __init__(self, lm, nclass, dropout=0.5):
        '''
        lm : STRING
            Langauage model name from hugging face
        '''

        super(BertClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(lm)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, nclass)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        with torch.no_grad():
          _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
    
class LSTM_Model(nn.Module):
    def __init__(self, lm, nclass):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(lm)
        self.hidden_size = self.bert.config.hidden_size
        self.LSTM = nn.LSTM(input_size= self.hidden_size, hidden_size=self.hidden_size, bidirectional=True)
        self.clf = nn.Linear(self.hidden_size*2, nclass)
        
    def forward(self, input_ids, masks):
        with torch.no_grad():
            #encoded_layers, pooled_
            output = self.bert(input_ids=input_ids, attention_mask= masks)
            encoded_layers, pooled_output = output[0], output[1]
        #print(output)
        encoded_layers = encoded_layers.permute(1, 0, 2)
        enc_hiddens, (last_hidden, last_cell) = self.LSTM(pack_padded_sequence(encoded_layers, masks))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output_hidden = F.dropout(output_hidden, 0.2)
        output = self.clf(output_hidden)
        
        return F.sigmoid(output)