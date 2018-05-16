import json
import pandas as pd
def convert_json_DF(f):
 #   with open(f) as train_file:
    dict_train = json.load(f)
    data_dict={}
    for k,v in zip(dict_train['header'],dict_train['data'][0]):
        data_dict[k]=v
    ddat=pd.DataFrame.from_dict(data_dict,orient='index')
    X_test_raw=ddat.T
    return X_test_raw


def prepare_data(df):
    import numpy as np
    from sklearn.preprocessing import FunctionTransformer
    sc = FunctionTransformer(np.log1p)
    X=df[['goal']]
    X= sc.transform(X)
    df[['goal']]=X
    df = pd.get_dummies(df, columns=['country'])
    df = pd.get_dummies(df, columns=['category'])
    df = pd.get_dummies(df, columns=['deadline_weekday'])
    df = pd.get_dummies(df, columns=['created_at_weekday'])
    df = pd.get_dummies(df, columns=['launched_at_weekday'])
    return df

def raw_final_DF(X_test_raw,DF_default):
    data=prepare_data(X_test_raw)
    DF_default.update(data)
    scale_mapper1 = {"FALSE":0, "TRUE":1}
    DF_default["staff_pick"]=DF_default["staff_pick"].astype(int)           
    return DF_default
