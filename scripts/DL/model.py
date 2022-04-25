# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:43:43 2022

@author: Cristina GH
"""


import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig
from torch.autograd import Variable



class SimpleBert(nn.Module):
    def __init__(self, lm, nclass, dropout=0.5):
        '''
        lm : STRING
            Langauage model name from hugging face
        '''

        super(SimpleBert, self).__init__()
        self.config = AutoConfig.from_pretrained(lm, num_labels=nclass)
        self.bert = AutoModelForSequenceClassification.from_pretrained(lm, config=self.config)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask): #_id, mask):
        #with torch.no_grad():
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)#_ids= input_id, attention_mask=mask, return_dict=False)
        return self.relu(pooled_output.logits)


class BertClassifier(nn.Module):

    def __init__(self, lm, nclass, dropout=0.5):
        '''
        lm : STRING
            Langauage model name from hugging face
        '''

        super(BertClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(lm, num_labels=nclass)
        self.bert = AutoModel.from_pretrained(lm, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, nclass)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        with torch.no_grad():
            _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
    
class LSTM(nn.Module):
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
        encoded_layers = encoded_layers.permute(1, 0, 2)
        lngt = [encoded_layers.shape[0]] * encoded_layers.shape[1]
        enc_hiddens, (last_hidden, last_cell) = self.LSTM(pack_padded_sequence(encoded_layers, lngt))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output_hidden = F.dropout(output_hidden, 0.2)
        output = self.clf(output_hidden)
        
        return F.sigmoid(output)
    

class BiLSTM_Attention(nn.Module):
    def __init__(self, lm, nclass):
        super(BiLSTM_Attention, self).__init__()
        self.bert = AutoModel.from_pretrained(lm)
        self.hidden_size = self.bert.config.hidden_size
        # self.token_embedding = {token: self.bert.get_input_embeddings()(torch.tensor(id))  for token, id in .get_vocab().items()}
        # print(len(self.token_embedding))
        # self.embedding = nn.Embedding(len(self.token_embedding), self.hidden_size)
        self.lstm = nn.LSTM(input_size= self.hidden_size, hidden_size=self.hidden_size, bidirectional=True)
        self.out = nn.Linear(self.hidden_size * 2, nclass)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_size * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, input_ids, masks):
        with torch.no_grad():
            output = self.bert(input_ids=input_ids, attention_mask= masks)
            encoded_layers, pooled_output = output[0], output[1]
        encoded_layers = encoded_layers.permute(1, 0, 2)
        
        lngt = [encoded_layers.shape[0]] * encoded_layers.shape[1]
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(pack_padded_sequence(encoded_layers, lngt))
        unpacked_o, unpacked_lengths = pad_packed_sequence(output, batch_first=True)
        #print(unpacked_o.shape)
        #output = unpacked_o.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(unpacked_o, final_hidden_state)
        return self.out(attn_output)#, attention 