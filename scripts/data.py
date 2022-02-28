# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 20:09:32 2022

@author: Cristina GH
"""
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm



def preprocess_data():
    pass


def prepare_data(train, test):
    '''
    
    Parameters
    ----------
    dev : TYPE
        DESCRIPTION.
    test : TYPE
        DESCRIPTION.

    Returns
    -------
    dataframes : TYPE
        DESCRIPTION.

    '''
    
    df_train = pd.read_csv(train)
    df_test = pd.read_csv(test)
    
    dataframes = {
      'train': df_train, 
      'test': df_test
    }
    
    for key, df in dataframes.items():
    
      columns_to_group_by_user = ['label', 'gender', 'profession', 'ideology_binary', 'ideology_multiclass']
    
      group = df.groupby(by = columns_to_group_by_user, dropna = False, observed = True, sort = False)
    
      # Custom df per user
      df_users = group[columns_to_group_by_user].agg(func = ['count'], as_index = False, observed = True).index.to_frame (index = False)
    
      merged_fields = []
    
      pbar = tqdm(df_users.iterrows(), total = df_users.shape[0], desc = "merging users")
        
      for index, row in pbar:
          df_user = df[(df['label'] == row['label'])]
          merged_fields.append({**row, **{field: ' [SEP] '.join (df_user[field].fillna ('')) for field in ['tweet']}})
        
      dataframes[key] = pd.DataFrame (merged_fields)
      
    return dataframes


      
      

