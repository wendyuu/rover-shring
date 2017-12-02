def color_df(df):
    df_black = df[df.all_requests_provider <= 2]
    df_green = df[(df.all_requests_provider <15) & (df.all_requests_provider >2)]
    df_red = df[(df.all_requests_provider >=15)]
    print 'red green black breakdown is '+ str(df_red.conversation_id.count()/(df.conversation_id.count()+0.0))+','+ str(df_green.conversation_id.count()/(df.conversation_id.count()+0.0))+','+ str(df_black.conversation_id.count()/(df.conversation_id.count()+0.0))
    return df_black,df_green,df_red
