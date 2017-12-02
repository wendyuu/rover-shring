def combine_df(df_orig,df_addon,key):
    '''combine two dataframes on key and ignore duplicate information from df_addon'''
    df = df_orig.merge(df_addon,on=key,suffixes=['','_dup'])
    for i in list(df):
        if 'dup' in i:
            df = df.drop(i,axis=1)
    return df
