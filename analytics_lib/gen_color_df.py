import pandas as pd
def gen_color_df(df,feature_col = 'all_requests_provider', threshold_new = 2, threshold_mature = 14):
    df_colored = df
    df_colored['status'] = None
    df_colored.loc[df_colored[feature_col] <= threshold_new,'status'] = 'seeding'
    df_colored.loc[(df_colored[feature_col] > threshold_new) & (df_colored[feature_col] <= threshold_mature),'status'] = 'green'
    df_colored.loc[(df_colored[feature_col] >= threshold_mature),'status'] = 'red'
    return df_colored
