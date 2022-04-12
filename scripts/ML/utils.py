# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 23:16:41 2022

@author: Cristina GH
"""

import data
import baseline

import time
import argparse
import pandas as pd


def submission_file(df, train_, X_test, labellist):
    '''
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    train_ : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    labellist : TYPE
        DESCRIPTION.
    save_predfile : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    save_location = '../../logs/'
    output_df = pd.DataFrame()
    output_df['user'] = df['test']['label']

    for label in labellist:#, 'profession', 'ideology_binary', 'ideology_multiclass']:
        output_df[label] = train_[label].predict(X_test)
    
    #print(output_df)
    output_df.rename(columns={'ideology_binary': "pib", 'ideology_multiclass': "pim"})

    #compression_opts = dict(method='zip',
     #                   archive_name='demoresults.csv', mtime=1)

    #output_df.to_csv('out.zip', compression=compression_opts, index=False)

    output_df.to_csv(f'{save_location}results_{time.time()}.csv')
    print('results saved!!!')
