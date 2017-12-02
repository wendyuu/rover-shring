def EvalRFModel(X,y,plot_name,no_bin=None,min_samples_leaf_no=None):
    
    from treeinterpreter import treeinterpreter as ti
    from sklearn.ensemble import RandomForestRegressor
    import analytics_lib
    from analytics_lib.Bucket_BR_Plot import Bucket_BR_Plot
    
    if no_bin is None:
        no_bin = 15

    if min_samples_leaf_no is None:
        min_samples_leaf_no = 100
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestRegressor( n_estimators = 200, oob_score = True, n_jobs = -1,random_state =50, max_features = "auto", min_samples_leaf = min_samples_leaf_no)
    rf.fit(X_train, y_train)
    predicted, bias, contributions = ti.predict(rf, X_test)  

    # generate class probabilities
    df = pd.DataFrame(predicted)
    df = df.join(y_test.reset_index(drop=True))
    df.columns = ['probability','booked']
    h = Bucket_BR_Plot(df,'probability','booked',no_bin,plot_name)
