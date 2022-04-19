# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 20:09:32 2022

@author: Cristina GH
"""
import argparse

import csv
import pandas as pd
import numpy as np
from tqdm import tqdm


def aggregate_users(df, name='train'):


    #########
    if name == 'testing':
        columns_to_group_by_user = ['label']#['label', 'gender', 'profession', 'ideology_binary', 'ideology_multiclass']
    else:
        columns_to_group_by_user = ['label', 'gender', 'profession', 'ideology_binary', 'ideology_multiclass']

    data_columns = ['tweet','clean_data','lemmatized_data','lemmatized_nostw', 'emojis']
    
    group = df.groupby(by = columns_to_group_by_user, dropna = False, observed = True, sort = False)
    df_users = group[columns_to_group_by_user].agg(func = ['count'], as_index = False, observed = True).index.to_frame(index = False)
    merged_fields = []
    pbar =  tqdm(df_users.iterrows(), total = df_users.shape[0], desc = "merging users")

    for index, row in pbar:
        df_user = df[(df['label'] == row['label'])]
        
        if len(df.columns) >  7:
            merged_fields.append({**row, **{field: ' [SEP] '.join(df_user[field].fillna('')) for field in data_columns}})
        else:
            merged_fields.append({**row, **{field: ' [SEP] '.join(df_user[field].fillna('')) for field in ['tweet']}})
    
    df = pd.DataFrame(merged_fields)
    #########
    '''
    columns_to_group_by_user = ['label', 'gender', 'profession', 'ideology_binary', 'ideology_multiclass']

    group = df.groupby(by = columns_to_group_by_user, dropna = False, observed = True, sort = False)

    # Custom df per user
    df_users = group[columns_to_group_by_user].agg(func = ['count'], as_index = False, observed = True).index.to_frame (index = False)

    merged_fields = []

    pbar = tqdm(df_users.iterrows(), total = df_users.shape[0], desc = "merging users")

    for index, row in pbar:
        df_user = df[(df['label'] == row['label'])]
        merged_fields.append({**row, **{field: ' [SEP] '.join (df_user[field].fillna ('')) for field in ['tweet']}})

    df = pd.DataFrame (merged_fields)
    '''
    return df


def prepare_data(train, test, testing=False):
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
    print('testing:', testing)
    try:
      df_train = pd.read_csv(train)
    except:
      df_train = pd.read_csv(train, sep='\t', quoting=csv.QUOTE_NONE)
    try:
      df_test = pd.read_csv(test)
    except:
      df_test = pd.read_csv(test, sep='\t', quoting=csv.QUOTE_NONE)

    print("trian col:", df_train.columns)
    print("test col:", df_test.columns)
    
    if testing:
        dataframes = {
          'train': df_train, 
          'testing': df_test
        }
    else:
        dataframes = {
          'train': df_train, 
          'test': df_test
        }
    # tweet aggregation
    new_dataframe = dict()
    for key, df in dataframes.items():
        if key == 'testing':
            new_dataframe['test'] = aggregate_users(df, name=key)
        else:
            new_dataframe[key] = aggregate_users(df, name=key)
    #print(dataframes)
    assert(len(new_dataframe) == 2)
    return new_dataframe


      
      

