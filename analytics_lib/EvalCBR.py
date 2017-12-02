def EvalCBR(feature_df,fea_list,booked_name,model_name,no_bin=None):
    fea_list = ['service_cbr_low_ci','provider_cbr_low_ci','service_cbr_low_ci_win','provider_cbr_low_ci_win','service_cbr_up_ci','provider_cbr_up_ci','service_cbr_up_ci_win','provider_cbr_up_ci_win','service_win_cbr','provider_win_cbr','service_cbr','provider_cbr']
    if no_bin == None:
        no_bin = 10
    y = feature_df[[booked_name]]
    df_auc = pd.DataFrame(columns=['cbr_name','auc'])
    
    for ind,cbr_info in enumerate(fea_list):
        X_cbr = feature_df[[cbr_info]]
        df = pd.merge(y,X_cbr,right_index=True, left_index=True)
        if (model_name == 'raw'):
            Bucket_BR_Plot(df,cbr_info,booked_name,no_bin,os.path.join(data_dir,'Figures',cbr_info+'_raw.png'))
        if (model_name == 'logit'):
            auc= EvalLRModel(X_cbr,y,os.path.join(data_dir,'Figures',cbr_info+'_logit.png'),no_bin)
        df_auc.loc[ind] = [cbr_info,auc]
    return df_auc
