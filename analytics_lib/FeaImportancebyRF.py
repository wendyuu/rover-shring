def FeaImportancebyRF(X,y,fig_name = 'Feature Importance Ranking'):
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    rf = RandomForestRegressor(n_estimators = 50)
    rf_fit = rf.fit(X, y)
    names = list(X)
    importances = rf_fit.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_fit.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances of the forest
    plt.figure(figsize=[17,17])
    plt.title(fig_name)
    importances = rf_fit.feature_importances_
    indices = np.argsort(importances)
   # print names[indices]
    plt.barh(range(len(indices)), importances[indices],  xerr=std[indices], color='r', align='center')

    
    plt.yticks(range(len(indices)), np.array(names)[indices])

    plt.xlabel('Relative Importance')
    plt.ylim([-1, X.shape[1]])
    plt.show()
    importance_df = pd.DataFrame(sorted(zip(map(lambda X: round(X, 4), rf_fit.feature_importances_), names), 
                 reverse=True),columns = ['importance','name'])

    importance_df['prevalance'] = np.zeros(len(importance_df.importance))
    for fea in list(X):
        pct = round((X[X[fea]>0][fea].count()+0.0)/(X[fea].count()+0.0),4)
        importance_df.loc[importance_df.name == fea,'prevalance'] = pct
    return importance_df
