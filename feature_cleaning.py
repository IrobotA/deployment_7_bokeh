import pandas as pd
import numpy as np



def nettoyage(df):
    for col_na in df.isna().sum()[df.isna().sum() > 0 ].index: 
        if df[col_na].dtype =='object':
            df[col_na] = df[col_na].fillna(df[col_na].mode().values[0]) # var categ val mqte => mot le plus fréquent
        else :
            df[col_na] = df[col_na].fillna(df[col_na].median()) # val mqtes => mediane
    return df.isna().sum().sum()
