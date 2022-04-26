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
import numpy as np
from tqdm import tqdm


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

from laserembeddings import Laser

def submission_file(df, train_, X_test, args):
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

    for label in args.runclass:#, 'profession', 'ideology_binary', 'ideology_multiclass']:
        output_df[label] = train_[label].predict(X_test)
    
    print(output_df)
    #output_df.rename(columns={'ideology_binary': "pib", 'ideology_multiclass': "pim"})

    #compression_opts = dict(method='zip',
     #                   archive_name='demoresults.csv', mtime=1)

    #output_df.to_csv('out.zip', compression=compression_opts, index=False)

    output_df.to_csv(f'{save_location}testresults_{args.model}_{args.feat}.csv', index=False)
    print('results saved!!!')


def get_tfidf(dfs, col='tweet'):
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
      max_features = 500,
      lowercase = True
    ) 
    
    X_train = vectorizer.fit_transform(dfs['train'][col])

    X_test = vectorizer.transform(dfs['test'][col])
    print("Data Loaded ...")
    return X_train, X_test



def get_ngramfeat(dfs, ngram=(1,3), score_type='count', col='tweet'):

    from stop_words import get_stop_words
    stop_words = get_stop_words('spanish')

    if score_type=='count':
        vectorizer = CountVectorizer(analyzer = 'word',ngram_range=ngram, stop_words=stop_words)
    else:
        vectorizer = TfidfVectorizer(analyzer = 'word',ngram_range=ngram, stop_words=stop_words)

    X_train = vectorizer.fit_transform(dfs['train'][col])

    X_test = vectorizer.transform(dfs['test'][col])
    print("Data Loaded ...")
    return X_train, X_test



def return_wvecs(train_tweets):

    #model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
    # add location to saved model
    model = Word2Vec.load("/home/amansinha/IberLEF_2022/notebook/word2vec_raw.model")

    x_train = []
    for sen in tqdm(train_tweets, desc="extracting features"):
        
        sens = []
        for ss in sen.split('[SEP]'):
            sens.extend(ss.split(' '))
        wvs = []
        for w in sens:
            if w in model.wv:
                wvs.append(model.wv[w])
            else:
                wvs.append(np.random.rand(100))
        wvs = np.asarray(wvs)
        x_train.append(np.mean(wvs, axis=0))
        
    x_train = np.asarray(x_train)
    return x_train

def return_lasers(train_tweets):

    laser = Laser()

    x_train = []
    for sen in tqdm(train_tweets, desc="extracting features"):
                
        embeds = laser.embed_sentences(sen.split('[SEP]')
                                ,lang='es')
        
        
        x_train.append(np.mean(embeds,axis=0))
    x_train = np.asarray(x_train)

    return x_train


def get_glovefeat(dfs, col='tweet'):

    X_train = return_wvecs(dfs['train'][col])
    X_test = return_wvecs(dfs['test'][col])

    print("Data Loaded ...")
    return X_train, X_test


def get_laserfeat(dfs, load_pretrained=True, col='tweet'):

    # load pretrained dfeats
    prefix = '/home/amansinha/IberLEF_2022/notebook/'

    if load_pretrained:
        print('Loading pretrained laser embedddings ...')
        X_train = np.load(prefix + 'xtrain_raw_laser.npy')
        X_test = np.load(prefix + 'xtest_raw_laser.npy')

    else:

        X_train = return_lasers(dfs['train'][col])
        X_test = return_lasers(dfs['test'][col])

    print("Data Loaded ...")
    return X_train, X_test