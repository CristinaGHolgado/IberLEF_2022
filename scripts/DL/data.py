# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:31:20 2022

@author: Cristina GH

BERT baseline
"""

import pandas as pd
import csv
import re
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class load_data:
    def __init__(self, filename):
        self.__filename = filename
        
    def load(self): 
        df = pd.read_csv(self.__filename, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
        df = df.dropna()
        
        labels = ['label', 'gender', 'profession','ideology_binary','ideology_multiclass']
        data_columns = ['tweet','clean_data','lemmatized_data','lemmatized_nostw', 'emojis']

        for col in data_columns:
            df[col] = df[col].astype(str) 

        df_grouped = df.groupby(labels)[data_columns].agg({lambda x: ' '.join(list(set(x)))}).reset_index()
        df_grouped.columns = df_grouped.columns.droplevel(1)
        df_grouped['emojis'] = df_grouped['emojis'].apply(lambda x: re.sub("\[|\]|'|,", '', x))
        
        return df_grouped



class split_data:
    def __init__(self, train_data, test_data):
        self._train_data = load_data(train_data).load()
        # self._test_data = load_data(test_data).load() 
    
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

    def __init__(self, df, lm, lclass):
        self.label_encoder = dict(zip(list(set(df.lclass)), list(range(len(df.lclass)))))
        self.labels = [self.label_encoder[label] for label in df[lclass]]
        # 'dccuchile/bert-base-spanish-wwm-cased'
        tokenizer = AutoTokenizer.from_pretrained(lm)
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
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
    
