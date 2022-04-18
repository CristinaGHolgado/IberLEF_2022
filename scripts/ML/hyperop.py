import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
pipe_svc = make_pipeline(imputer,StandardScaler(),PCA(n_components=2),SVC(random_state=1))
param_range = [0.001,0.01,0.1,1,10,100,1000]
param_grid = {'svc__C': [0.001,0.01,0.1,1,10,100,1000], 'svc__kernel': ['linear', 'rbf'],
              'svc__gamma': [0.001,0.01,0.1,1,10,100,1000]}
cv = StratifiedKFold(n_splits=5)
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid, scoring='accuracy', cv = cv,
                  return_train_score=True)
gs.fit(X_train, y_train)

print("Best Estimator: \n{}\n".format(gs.best_estimator_))
print("Best Parameters: \n{}\n".format(gs.best_params_))
print("Best Test Score: \n{}\n".format(gs.best_score_))
print("Best Training Score: \n{}\n".format(gs.cv_results_['mean_train_score'][gs.best_index_]))
print("All Training Scores: \n{}\n".format(gs.cv_results_['mean_train_score']))
print("All Test Scores: \n{}\n".format(gs.cv_results_['mean_test_score']))
# # This prints out all results during Cross-Validation in details
#print("All Meta Results During CV Search: \n{}\n".format(gs.cv_results_))




"""
models= ['svm', 'mlp', 'sgd', 'lgr', 'rf', 'nb']


svm :
kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
gamma{‘scale’, ‘auto’}
class_wieght 

mlp

activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
solver{‘lbfgs’, ‘sgd’, ‘adam’}
learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}
momentum for sgd  solver: between 0 and 1
nesterovs_momentum for sgd solver: between 0 and 1
early_stopping : True (validation_fraction)
beta_1, beta_2 for adam solver


sgd :

loss option : ‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, or a regression loss: ‘squared_error’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
penalty{‘l2’, ‘l1’, ‘elasticnet’}
early_stopping 
validation_fraction
class_weight

lgr
penalty{‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
class_weight
solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}

rf
criterion{“gini”, “entropy”}
max_features{“auto”, “sqrt”, “log2”}

nb



"""