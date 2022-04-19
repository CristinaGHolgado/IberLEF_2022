# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 23:16:51 2022

@author: Cristina GH
"""
import argparse
import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import data
import utils


def train(args, data, X_train, X_test):
    '''
    

    Parameters
    ----------
    args : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    labels = args.runclass#['gender']#, 'profession', 'ideology_binary', 'ideology_multiclass']
    baselines = {}
    
    f1_scores = {}

    for label in labels:

        if args.model == 'svm':
            model = OneVsRestClassifier(svm.SVC(gamma=0.01, 
                                                       C=50., 
                                                       probability=True, 
                                                       class_weight='balanced', 
                                                       kernel='linear', cache_size = 4000))
        elif args.model == 'mlp':
            model = MLPClassifier(max_iter=500)
        elif args.model == 'sgd':
            model = SGDClassifier(max_iter=1000, tol=1e-3)
        elif args.model == 'rf':
            model = RandomForestClassifier(max_depth=2)
        elif args.model == 'lgr':
            model = LogisticRegression()#class_weight='balanced')
        elif args.model == 'nb':
            model = MultinomialNB()
        elif args.model == 'xgb':
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1)

        baselines[label] = model
        # 
        # SGDClassifier(max_iter=1000, tol=1e-3)
        # 
        # LogisticRegression()#class_weight='balanced')
        print(f'\nRunning {args.model} ...\n'); t1 = time.time()
        baselines[label].fit(X_train, data['train'][label])
        print(f"Model fitted : {time.time()-t1:.4f} seconds")
        y_pred = baselines[label].predict(X_test)
        
        if label in data['test'].columns:
            class_report = classification_report(data['test'][label], 
                                                  y_pred, 
                                                  zero_division = 0, 
                                                  digits = 6, 
                                                  output_dict=True)
            
            f1_scores[label] = f1_score(data['test'][label], y_pred, average='macro')
            
            class_report_df = pd.DataFrame(class_report).transpose()
            # class_report_df.to_csv(str(f"classification_report_{label}.tsv"), sep='\t', encoding='utf-8')
            
            print(label.upper())
            print(class_report_df)

            # error - analysis
            if args.doea:
                error_indexes  = y_pred != data['test'][label]
                error_df = pd.DataFrame({'gold': data['test'][label][error_indexes].values, 'tweet': data['test']['tweet'][error_indexes].values, 'pred': y_pred[error_indexes]})
                if len(error_df) >1:
                    error_df.to_csv(f'~/IberLEF_2022/logs/{args.model}_{label}_errors.tsv',sep='\t', index = False)

    print(set(labels) , set(data['test'].columns))
    if len(set(set(labels) & set(data['test'].columns)))>1:
        print(set(labels) , set(data['test'].columns))
        f1_scores = list(f1_scores.values())
        print ("\nOVERALL F1-score : {f1}".format(f1 = sum(f1_scores) / len(f1_scores)))
    else:
        f1_scores = [-1]
    #print("real testing ...")
    #utils.submission_file(data, baselines, X_test, args)
    #print(f'saved for {label}!!')

    
    return baselines, f1_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', '--train_file', required=True, help="Path to training data")
    parser.add_argument('-test_file', '--test_file', required=True, help="Path to test data")
    parser.add_argument('-model', '--model', required = True, help='Name of classifier model', choices=['svm', 'mlp', 'sgd', 'lgr', 'rf', 'nb', 'xgb'])
    parser.add_argument('-feat', '--feat', required = True, help='Name of feature to extract', choices=['glove', 'laser', 'ngram', 'tfidf'])
    parser.add_argument('-runclass', '--runclass' , nargs='+', help="Run classifier for class", default = ['gender' ,'profession' ,'ideology_binary', 'ideology_multiclass'])
    parser.add_argument('-doea', '--doea', action='store_true' , help='Perform error analysis files')
    parser.add_argument('-testing', '--testing', action='store_true' , help='Perform prediction on test file with no labels')
    parser.add_argument('-save_pred', '--save_pred', action='store_true', help='Saving prediction on test set')
    parser.add_argument('-REPEAT', '--REPEAT', default=1, type=int, help='repeat the experiment')
    parser.add_argument('-inpcol', '--inpcol', default='tweet', help='choose preprocessed column', choices=['tweet', 'clean_data','lemmatized_data', 'lemmatized_nostw', 'emojis'])
    args = parser.parse_args() 

    _data = data.prepare_data(args.train_file, args.test_file, testing=args.testing)

    if args.feat == 'glove':
        X_train, X_test = utils.get_glovefeat(_data, col=args.inpcol)
    elif args.feat == 'laser':
        X_train, X_test = utils.get_laserfeat(_data, col=args.inpcol)
    elif args.feat == 'ngram':
        X_train, X_test = utils.get_ngramfeat(_data, col=args.inpcol)
    elif args.feat == 'tfidf':
        X_train, X_test = utils.get_tfidf(_data, col=args.inpcol)

    print(X_train.shape, X_test.shape)
    f1s = []
    for i in range(args.REPEAT): 
        model, _  = train(args, _data, X_train, X_test)
        f1s.extend(_)

    if args.save_pred:
        output = utils.submission_file(_data, model, X_test, args)

    if sum(_) <= -1:
        print('testing.....')
        output = utils.submission_file(_data, model, X_test, args)

    if args.REPEAT > 1:
        print(f'For {args.runclass};  mean f1: {np.mean(f1s)}')
    else:
        print(*f1s , sep='|')


    # cross validation
    #y_train = _data['train'][args.runclass].values.ravel()
    #from sklearn.model_selection import cross_val_score, cross_validate
    #print(svm_train[args.runclass[0]])
    #scores = cross_validate(svm_train[args.runclass[0]], X_train, y_train, cv=5, scoring='f1_macro')
    #print(scores)
    #print("%0.2f accuracy with a standard deviation of %0.2f" % (scores['test_score'].mean(), scores['test_score'].std()))
