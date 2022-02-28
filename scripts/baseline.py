# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 23:16:51 2022

@author: Cristina GH
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


def vectorize(dfs):
    '''
    

    Parameters
    ----------
    dfs : TYPE
        DESCRIPTION.

    Returns
    -------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.

    '''
    vectorizer = TfidfVectorizer(
      analyzer = 'word',
      min_df = .1,
      max_features = 5000,
      lowercase = True
    ) 
    
    X_train = vectorizer.fit_transform(dfs['train']['tweet'])

    X_test = vectorizer.transform(dfs['test']['tweet'])
    
    return X_train, X_test


def train(data, X_train, X_test):
    '''
    

    Parameters
    ----------
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
    
    labels = ['gender', 'profession', 'ideology_binary', 'ideology_multiclass']
    baselines = {}
    
    f1_scores = {}

    for label in labels:

        baselines[label] = OneVsRestClassifier(svm.SVC(gamma=0.01, 
                                                       C=50., 
                                                       probability=True, 
                                                       class_weight='balanced', 
                                                       kernel='linear'))

        # MLPClassifier(random_state=1, max_iter=500)
        # SGDClassifier(max_iter=1000, tol=1e-3)
        # RandomForestClassifier(max_depth=2, random_state=0)
        # LogisticRegression()#class_weight='balanced')

        baselines[label].fit(X_train, data['train'][label])
        
        y_pred = baselines[label].predict(X_test)
        
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
        print()

    f1_scores = list(f1_scores.values())
    print ("F1-score : {f1}".format(f1 = sum(f1_scores) / len(f1_scores)))
    
    return baselines


