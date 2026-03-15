from data_loader import load_raw_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def split(output_col_list):
    X,y=load_raw_data(output_col_list)
    X_train,X_test,y_train,y_test=train_test_split(X,y,shuffle=False,random_state=42)
    return X_train,X_test,y_train,y_test
    
