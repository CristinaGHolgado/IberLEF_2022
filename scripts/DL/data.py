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
import tqdm

def load_data(file, agg=1):      
    with open(file, 'r', encoding='utf-8') as csvfile:
        
        dialect = csv.Sniffer().sniff(csvfile.readline()) # detect delimiter
        
        df = ''
        
        try:
            df = pd.read_csv(file, sep=str(dialect.delimiter))
        
        except pd.errors.ParserError:
            df = pd.read_csv(file, sep=str(dialect.delimiter), quoting=csv.QUOTE_NONE)

        if agg == 0: 
            # without aggregating
            return df
            
        columns_to_group_by_user = ['label', 'gender', 'profession', 'ideology_binary', 'ideology_multiclass']
        data_columns = ['tweet','clean_data','lemmatized_data','lemmatized_nostw', 'emojis']
        
        group = df.groupby(by = columns_to_group_by_user, dropna = False, observed = True, sort = False)
        df_users = group[columns_to_group_by_user].agg(func = ['count'], as_index = False, observed = True).index.to_frame(index = False)
        merged_fields = []
        pbar =  tqdm.tqdm(df_users.iterrows(), total = df_users.shape[0], desc = "merging users")

        for index, row in pbar:
            df_user = df[(df['label'] == row['label'])]
            
            if len(df.columns) >  7:
                merged_fields.append({**row, **{field: ' [SEP] '.join(df_user[field].fillna('')) for field in data_columns}})
            else:
                merged_fields.append({**row, **{field: ' [SEP] '.join(df_user[field].fillna('')) for field in ['tweet']}})
        
        df = pd.DataFrame(merged_fields)
        
        return df



# class split_data:
#     def __init__(self, train_data, test_data):
#         self._train_data = load_data(train_data).load()
#         # self._test_data = load_data(test_data).load() 
    
#     def _split(self):
#         np.random.seed(112)
#         df_train, df_val, df_test = np.split(self._train_data.sample(frac=1, random_state=42), 
#                                          [int(.8*len(self._train_data)), int(.9*len(self._train_data))])
#         print(len(df_train),len(df_val), len(df_test))
        
#         return df_train, df_val, df_test



class spanish_dataset(torch.utils.data.Dataset):

    def __init__(self, df, lm, lclass):
        self.label_encoder = dict(zip(list(set(df[lclass])), list(range(len(df[lclass])))))
        self.labels = [self.label_encoder[label] for label in df[lclass]]
        # 'dccuchile/bert-base-spanish-wwm-cased'
        tokenizer = AutoTokenizer.from_pretrained(lm)
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length=128, truncation=True,
                                return_tensors="pt") for text in df['clean_data']]

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
    
