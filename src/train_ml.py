from split import split
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.ensemble import GradientBoostingClassifier,RandomForestRegressor,StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from preprocess import get_preprocessor
from data_loader import load_raw_data
from split import split

def train_ml(output_col_list):
    X_train,X_test,y_train,y_test=split(output_col_list)
    preprocessor=get_preprocessor(X_train)
    pipeline=Pipeline([
        ("preprocessor",preprocessor),
        ("model",LinearRegression())
    ])
    
    pipeline.fit(X_train,y_train)
    y_pred=pipeline.predict(X_test)
    r2=r2_score(y_test,y_pred)
    print('R2 scoree is ',r2)
    return pipeline

if __name__=='__main__':
    output_col_list=['casual','registered','cnt']
    model=train_ml(output_col_list)
    

