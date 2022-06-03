#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import csv

import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.utils import class_weight



from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# In[218]:


train_df = pd.read_csv('../new_data/train_lemmatized_nostw_newsplit.csv', sep='\t')
dev_df = pd.read_csv('../new_data/dev_lemmatized_nostw_newsplit.csv',sep='\t')
test_df = pd.read_csv('../new_data/devtest_lemmatized_nostw_newsplit.csv',sep='\t')
testt_df = pd.read_csv('../data/preprocessed_test_without_labels.csv', sep='\t', quoting=csv.QUOTE_NONE)

print('loaded DataFrames')

new_df = pd.concat([train_df, dev_df, test_df])
new_df = new_df.drop_duplicates(subset=['tweet'])

print('loaded integrateddf ')

def aggregate_users(df, testing = False):
    if not testing:
        columns_to_group_by_user = ['label', 'gender', 'profession', 'ideology_binary', 'ideology_multiclass']
    else:
        columns_to_group_by_user = ['label']

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
    return df

train_df = aggregate_users(train_df)
dev_df = aggregate_users(dev_df)
test_df = aggregate_users(test_df)
testt_df = aggregate_users(testt_df, testing=True)


agg_df = aggregate_users(new_df)

print('aggregate_users done!!!!')

from stop_words import get_stop_words
stop_words = get_stop_words('spanish')

# In[140]:

prefix = '/home/amsinha/IberLEF_2022/notebook/'
train_tfidf500 = np.load(prefix + 'lemm_nostw_train_tfidf500.npy', allow_pickle=True)
dev_tfidf500 = np.load(prefix + 'lemm_nostw_dev_tfidf500.npy', allow_pickle=True)
test_tfidf500 = np.load(prefix + 'lemm_nostw_test_tfidf500.npy', allow_pickle=True)
testt_tfidf500 = np.load(prefix + 'lemm_nostw_testt_tfidf500.npy', allow_pickle=True)

train_cv500 = np.load(prefix + 'lemm_nostw_train_cv500.npy', allow_pickle=True)
dev_cv500 = np.load(prefix + 'lemm_nostw_dev_cv500.npy', allow_pickle=True)
test_cv500 = np.load(prefix + 'lemm_nostw_test_cv500.npy', allow_pickle=True) 
testt_cv500 = np.load(prefix + 'lemm_nostw_testt_cv500.npy', allow_pickle=True)
#np.load(prefix + '('clean_data_tfidf500.npy', Xtfidf)
#np.load(prefix + '('clean_data_cv500.npy', Xcv)

#np.load(prefix + '('clea_data_laser.npy', xlaser)
slaser_trdf = np.load(prefix + 'cleandata_train_laser.npy', allow_pickle=True) 
slaser_devdf = np.load(prefix + 'cleandata_dev_laser.npy', allow_pickle=True) 
slaser_testdf = np.load(prefix + 'cleandata_test_laser.npy', allow_pickle=True) 
slaser_testt_df = np.load(prefix + 'cleandata_testt_laser.npy', allow_pickle=True)


#np.load(prefix + '('lemm_nostw_gloveavg.npy', xgloveavg)
#np.load(prefix + '('lemm_nostw_gloveall.npy', xgloveall)
train_gloveavg  = np.load(prefix + 'lemm_nostw_gloveavg_trdf.npy', allow_pickle=True)
#train_gloveall = np.load(prefix + 'lemm_nostw_gloveall_trdf.npy', allow_pickle=True) 
dev_gloveavg = np.load(prefix + 'lemm_nostw_gloveavg_dvdf.npy', allow_pickle=True)
#dev_gloveall = np.load(prefix + 'lemm_nostw_gloveall_dvdf.npy', allow_pickle=True) 
test_gloveavg = np.load(prefix + 'lemm_nostw_gloveavg_tsdf.npy', allow_pickle=True) 
#test_gloveall = np.load(prefix + 'lemm_nostw_gloveall_tsdf.npy', allow_pickle=True)
testt_gloveavg = np.load(prefix + 'lemm_nostw_gloveavg_tt_df.npy', allow_pickle=True) 
#testt_gloveall = np.load(prefix + 'lemm_nostw_gloveall_ttdf.npy', allow_pickle=True) 


#np.load(prefix + '('emojifeat.npy', evecs)
tr_evecs = np.load(prefix + 'emojifeat_trdf.npy', allow_pickle=True)
dv_evecs = np.load(prefix + 'emojifeat_dvdf.npy', allow_pickle=True)
ts_evecs = np.load(prefix + 'emojifeat_tsdf.npy', allow_pickle=True) 
tt_evecs = np.load(prefix + 'emojifeat_ttdf.npy', allow_pickle=True)


tr_topics = pd.read_csv('../data/topics_ldamallet/train_topvect_newsplit.tsv', sep='\t').iloc[:,5:].to_numpy()
dv_topics = pd.read_csv('../data/topics_ldamallet/dev_topicsvect_newsplit.tsv', sep='\t').iloc[:,5:].to_numpy()
ts_topics = pd.read_csv('../data/topics_ldamallet/devtest_topvect_newsplit.tsv', sep='\t').iloc[:,5:].to_numpy()
tt_topics = pd.read_csv('../data/topics_ldamallet/test_nolabels_topvect_newsplit.tsv', sep='\t').iloc[:,1:].to_numpy()


#print(train_tfidf500.shape, dev_tfidf500)#np.concatenate([train_tfidf500, dev_tfidf500], axis=0).shape)
print("DATA LOADED !!!!!!")

features = {
    'tfidf' :(np.concatenate([train_tfidf500, dev_tfidf500], axis=0), test_tfidf500, testt_tfidf500),
    'cv' : (np.concatenate([train_cv500, dev_cv500], axis=0), test_cv500, testt_cv500),
    'emojis' :(np.concatenate([tr_evecs, dv_evecs], axis=0), ts_evecs, tt_evecs),
    'glove':(np.concatenate([train_gloveavg, dev_gloveavg], axis=0), test_gloveavg, testt_gloveavg),
    'laser':(np.concatenate([slaser_trdf, slaser_devdf], axis=0), slaser_testdf, slaser_testt_df),
    'topics' : (np.concatenate([tr_topics, dv_topics], axis=0), ts_topics, tt_topics)
}

DATA = {
    'train' : pd.concat([train_df, dev_df]),
    'test1': test_df,
    'test2': testt_df
}

#print(pipe_model.get_params().keys())

param_range = [0.001,0.01,0.1,1,10,100,1000]

param_grid_svc = {
                'svc__C': [0.001,0.01,0.1,1,10,100,1000], 
              'svc__kernel': ['linear', 'rbf'],
              'svc__gamma': [0.001,0.01,0.1,1,10,100,1000]
             }#98

param_grid_nb = {
    'multinomialnb__fit_prior' : [True, False],
    'multinomialnb__alpha' : [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
}#5

param_grid_mlp = {
    'mlpclassifier__activation' : ['logistic', 'tanh', 'relu'], 
    'mlpclassifier__alpha': [0.001, 0.0001, 0.1], 
    'mlpclassifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'mlpclassifier__solver': ['lbfgs', 'sgd', 'adam']
}#81

param_grid_rf = {'randomforestclassifier__class_weight' : ['balanced', 'balanced_subsample', None], 
                 'randomforestclassifier__criterion': ['gini', 'entropy'], 
                 'randomforestclassifier__max_features':['auto', 'sqrt', 'log2'], 
                 'randomforestclassifier__n_estimators' : [100, 200, 300], 
                 }#72

param_grid_sgd = {
    'sgdclassifier__class_weight': [None, 'balanced'], 
    'sgdclassifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'sgdclassifier__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],  
    'sgdclassifier__penalty': ['l2', 'l1', 'elasticnet'],
}#90


param_grid_xgb = {
                  'gradientboostingclassifier__learning_rate': [0.1, 0.01, 0.001], 
                  'gradientboostingclassifier__loss':['deviance'], 
                  'gradientboostingclassifier__max_depth':[1,2,3], 
                  'gradientboostingclassifier__max_features':['sqrt', 'log2'],
                  'gradientboostingclassifier__n_estimators': [100, 500],
                  }#72

config = {
        'svm': (SVC(random_state=1),param_grid_svc),
        'nb':(MultinomialNB(fit_prior=True), param_grid_nb),
        'mlp': (MLPClassifier(random_state=1, max_iter=2000, early_stopping=True, hidden_layer_sizes=1000),param_grid_mlp),
        'xgb': (GradientBoostingClassifier(n_estimators=100, random_state=0), param_grid_xgb),
        'sgd': (SGDClassifier(max_iter=5000, eta0=0.001,tol=1e-3, random_state=0, early_stopping=True), param_grid_sgd),
        'rf': (RandomForestClassifier(max_depth=2, random_state=0), param_grid_rf)
}



def run_hyparamsearch(pipe_model, param_grid, trainfeat, train_df, test1feat, test1_df, test2feat, name, logfile, resample=False):

    labels = ['gender','profession','ideology_binary', 'ideology_multiclass']
    print(pipe_model.get_params().keys(), file=logfile)

    ans_df = pd.DataFrame()
    ans_df['label'] = testt_df.label


    for label in labels:
        print("Curent label __", label, file=logfile)
        print("*"*50, file=logfile)
        cv = StratifiedKFold(n_splits=5)
        gs = GridSearchCV(estimator=pipe_model,
                          param_grid=param_grid, 
                          scoring='f1_weighted', cv = cv,
                          return_train_score=True, n_jobs=-1)
        #gs.fit(np.concatenate([train_tfidf500.A], axis=1), train_df[label])
        if resample:
            X_resampled, y_resampled = SMOTE().fit_resample(trainfeat,train_df[label])
            print(sorted(Counter(y_resampled).items()))
            gs.fit(X_resampled, y_resampled)
        else:
            gs.fit(trainfeat,train_df[label])

        print("Best Estimator: \n{}\n".format(gs.best_estimator_), file=logfile)
        print("Best Parameters: \n{}\n".format(gs.best_params_), file=logfile)
        print("Best Test Score: \n{}\n".format(gs.best_score_), file=logfile)
        print("Best Training Score: \n{}\n".format(gs.cv_results_['mean_train_score'][gs.best_index_]), file=logfile)
        print("All Training Scores: \n{}\n".format(gs.cv_results_['mean_train_score']), file=logfile)
        print("All Test Scores: \n{}\n".format(gs.cv_results_['mean_test_score']), file=logfile)



        ### Dev Test ####
        print('#'*10, "Development Test", '#'*20, file=logfile)
        preds = gs.best_estimator_.predict(test1feat)
        cm =confusion_matrix(test1_df[label], preds)
        cr = classification_report(test1_df[label], preds, zero_division = 0)
        print(cm, file=logfile)
        print(cr, file=logfile)

        ###### Generating labels #######
        print('#'*10, "Generating labels", '#'*20, file=logfile)
        testt_preds = gs.best_estimator_.predict(test2feat)
        ans_df[label] = testt_preds
    
    ans_df.to_csv(f'/home/amsinha/IberLEF_2022/logs/new_logs/{name}.csv', index=False)
    print(f'prediction for {name} saved!')

    print("analyze", file=logfile)
    for label in labels:
        ldict=dict()
        for a in ans_df[label]:
            if a in ldict:
                ldict[a] +=1
            else:
                ldict[a] = 1
        print(ldict, file=logfile)

    print("#"*40, file=logfile)

    return ans_df


if __name__ == '__main__':

    import time
    save_dir = '/home/amsinha/IberLEF_2022/logs/new_logs/'
    
    for model_name in [ 'mlp']: 

        if model_name != 'mlp':
            continue
        t1 = time.time(); print(f"{model_name} started at {t1}")
        for feat_name in tqdm(['topics'], desc=f'{model_name} running ...\n') : 
            if model_name == 'nb':
                if feat_name not in ['tfidf', 'cv']:
                    print('combination skipeed : ', model_name , '+', feat_name)
                    continue
            name = f'{model_name}_{feat_name}'
            f = open(save_dir + 'LOG_' + name+'.txt', 'w')
            print('#'*20, name, '#'*20, file=f)

            pipe_model, param_grid = config[model_name]

            trainfeat, test1feat, test2feat = features[feat_name]
            _train_df, test1_df, test2_df = DATA.values()

            ans_df = run_hyparamsearch(make_pipeline(pipe_model), 
                param_grid, trainfeat, _train_df, test1feat, test1_df, test2feat,
                name = name, logfile = f)

            t2 = time.time()
            print(f'Time taken {t2-t1:.4f} seconds')
            print(name,' IS DONE !!')
            f.close()
            
        
    

