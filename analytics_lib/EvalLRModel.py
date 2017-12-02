# evaluate the Logistic Regression model by splitting into train and test sets
def EvalLRModel(X,y,plot_name,penalty='l1',class_weight='balanced',no_bin=None,ylim=None):
    import analytics_lib
 #   from analytics_lib.Bucket_BR_Plot import Bucket_BR_Plot
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20,stratify=y)
    model2 = LogisticRegression(penalty=penalty,class_weight=class_weight)
    model2.fit(X_train, y_train)
    
    if no_bin is None:
        no_bin = 15
    # predict class labels for the test set
    predicted = model2.predict(X_test)

    # generate class probabilities
    probs = model2.predict_proba(X_test)
    df=pd.DataFrame(probs[:,1])
    df = df.join(y_test.reset_index(drop=True))
    df.columns = ['probability','booked']
    h = Bucket_BR_Plot(df,'probability','booked',no_bin,plot_name, ylim) 
    
    fpr, tpr, _ = metrics.roc_curve(y_test, probs[:,1])
    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    plt.plot(df.fpr,df.tpr)
    plt.title(os.path.basename(plot_name).replace('.png','_ROC'))
    auc = metrics.roc_auc_score(y_test, probs[:,1])
    plt.text(0.5, 0.5,'roc auc is '+ str(auc),
     horizontalalignment='left',
     verticalalignment='center',
     )
    plt.show()
    
    print 'roc auc is ', metrics.roc_auc_score(y_test, probs[:,1])
    
    print metrics.classification_report(y_test, predicted)
    
    return auc
