def get_training_conversation(df,exclude = True,service=None):
    '''get training dataset that exludes '''
    if(exclude == True):
        df_training = df[(df.convo_is_premier==0 )&(df.is_repeat_customer==False) &(df['ignore_from_search_reason'].isnull() == True) &(df.came_from_search == True)]
    else:
        df_training = df
    if service != None:
        df_training = df_training[df_training.service_type == service]
    return df_training
