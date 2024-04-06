import os.path

import pandas as pd
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.model_selection import train_test_split
import time


def rf(X,y):
    start = time.time()
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=1)
    print(X_train.shape,X_val.shape)
    print(y_train.shape,y_val.shape)

    model = RF(max_depth=100,n_estimators=100)
    model.fit(X_train,y_train)

    perm = PermutationImportance(model,random_state=1).fit(X_val,y_val)
    weights = eli5.explain_weights(perm,feature_names=X_val.columns.tolist(), top=400)
    df = eli5.format_as_dataframe(weights)
    path = './data/Random_Forest_Data/result/'
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv('./data/Random_Forest_Data/result/all_rank.csv', index=None)
    end = time.time()

    print('random_forest_feature_importance cost time: %.4f min.' % ((end-start)/60))