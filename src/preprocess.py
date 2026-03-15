import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



def get_preprocessor(X):
    col_info=pd.DataFrame({
        "dtype":X.dtypes,
        "nunique":X.nunique()
    })

    numerical_cols=col_info[col_info['nunique']>13].index
    categorical_cols=col_info[col_info['nunique']<=13].index

    # Pipelines for numerical and categorical columns
    num_pipeline=Pipeline([
        ('Scaler',StandardScaler())
    ])
    cat_pipeline=Pipeline([(
        "encoder",OneHotEncoder(drop='first',handle_unknown="ignore",sparse_output=False)
    )])

    #Collated ColumnTransformer
    preprocessor=ColumnTransformer([
        ("num",num_pipeline,numerical_cols),
        ("cat",cat_pipeline,categorical_cols)
    ])
    
    return preprocessor


