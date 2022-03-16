# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 23:16:41 2022

@author: Cristina GH
"""

import data
import baseline

import argparse
import pandas as pd


def submission_file(df, train_, X_test, labellist, save_predfile=False):
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
    output_df = pd.DataFrame()
    output_df['user'] = df['test']['label']

    for label in labellist:#, 'profession', 'ideology_binary', 'ideology_multiclass']:
        output_df[label] = train_[label].predict(X_test)
    
    #print(output_df)
    if save_predfile:
        output_df.to_csv('results.csv', index = False)
    