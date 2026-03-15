# $$
import pandas as pd
def load_raw_data(output_col_list=None,path='data/raw/day.csv'):
    if output_col_list==None:
        output_col_list=['casual','registered','cnt']
    df=pd.read_csv(path)
    df.drop(columns=['instant','dteday'],inplace=True)
    X=df.drop(columns=output_col_list)
    y=df[output_col_list]
    return X,y

if __name__=='__main__':
    output_col_list=['casual','registered','cnt']
    X,y=load_raw_data(output_col_list)
    print("Input shape:",X.shape)
    print('Output shape',y.shape)
# $$