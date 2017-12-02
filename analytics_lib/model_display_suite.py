def display_model_coef(feature_name, model,display_switch=True):
    from IPython.display import display
    import pandas as pd
    list1 = ['intercept']+feature_name
    list2 = list(model.intercept_) + list(model.coef_[0])
    df = pd.DataFrame(zip(list1,list2))
    if (display_switch == True):
        display(df)
    return df
