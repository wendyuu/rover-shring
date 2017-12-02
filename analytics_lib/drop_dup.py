import pandas as pd
def drop_dup(df,dup):
    for i in list(df):
        if dup in i:
            print dropping + i
            df = df.drop(i,axis=1)
    return df
