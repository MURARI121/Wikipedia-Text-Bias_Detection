#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier


# In[ ]:


train = pd.read_csv("Input_Data/text_processed2.csv")
train_encoded = pd.read_pickle('Input_Data/train_encoded2.pickle')
truth=pd.read_csv('Input_Data/ground_truth2.csv')


# In[ ]:


params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'reg_alpha':[1e-5, 1e-2, 0.1, 1,10],
        'learning_rate' :[0.01,0.05,0.1,0.2],
        'n_estimators':[500,750,1000,1500]
        }


# In[ ]:


xgb = XGBClassifier()


# In[ ]:


trainset=pd.DataFrame()
testset=pd.DataFrame()


# In[ ]:


i=1051552
trainset=train_encoded[:i]
testset=train_encoded[i:]
word=train_encoded['text'][i:]
y_train=trainset['label']
y_test=testset['label']
ID=testset['id']


# In[ ]:


trainset = trainset.drop(['id','label','text'],axis=1)
testset = testset.drop(['id','label','text'],axis=1)


# In[ ]:


from sklearn.metrics import recall_score


# In[ ]:


folds = 5
param_comb = 1000

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 100)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='recall', n_jobs=4, cv=skf.split(trainset, y_train), verbose=3, random_state=100 )
#grid = GridSearchCV(estimator=xgb, param_grid=params, scoring='recall', n_jobs=4, cv=skf.split(trainset, y_train), verbose=3 )
#grid.fit(trainset, y_train)

random_search.fit(trainset, y_train)


# In[ ]:




print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('xgb-random-grid-search-results-01.csv', index=False)


# In[ ]:




