# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:31:20 2022

@author: Cristina GH

BERT baseline
"""

import pandas as pd
import csv
import argparse
import numpy as np
import torch
from transformers import BertTokenizer


class load_data:
    def __init__(self, filename):
        self.__filename = filename
        
    def load(self): 
        df = pd.read_csv(self.__filename)
        df = df.dropna()
        df['tweet'] = df['tweet'].str.replace('@user','')
        df = df[df.tweet.str.len() > 10]
        return df 



class split_data:
    def __init__(self, train_data, test_data):
        self._train_data = load_data(train_data).load()
        self._test_data = load_data(test_data).load()
    
    def _split(self):
        np.random.seed(112)
        df_train, df_val, df_test = np.split(self._train_data.sample(frac=1, random_state=42), 
                                         [int(.8*len(self._train_data)), int(.9*len(self._train_data))])
        print(len(df_train),len(df_val), len(df_test))
        
        return df_train, df_val, df_test



labels = {'left':0,
          'moderate_left':1,
          'right':2,
          'moderate_right':3}

class spanish_dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [labels[label] for label in df['ideology_multiclass']]
        tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=False,
                                return_tensors="pt") for text in df['tweet']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
    
