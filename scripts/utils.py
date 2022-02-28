# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 23:16:41 2022

@author: Cristina GH
"""

import data
import baseline

import argparse
import pandas as pd


def submission_file(df, train_, X_test):
    '''
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    train_ : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    output_df = pd.DataFrame()
    output_df['user'] = df['test']['label']

    for label in ['gender', 'profession', 'ideology_binary', 'ideology_multiclass']:
        output_df[label] = train_[label].predict(X_test)
    
    print(output_df)
    # output_df.to_csv('results.csv', index = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', '--train_file', required=True, help="Path to training data")
    parser.add_argument('-test_file', '--test_file', required=True, help="Path to test data")
    args = parser.parse_args() 

    _data = data.prepare_data(args.train_file, args.test_file)
    X_train, X_test = baseline.vectorize(_data)
    svm_train = baseline.train(_data, X_train, X_test)
    output = submission_file(_data, svm_train, X_test)
    