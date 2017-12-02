import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

#plt.style.use('seaborn')
from IPython.core.pylabtools import figsize
figsize(16,8)

np.seterr(divide='ignore', invalid='ignore')
pd.options.mode.chained_assignment = None  # default='warn'

def plot_two(data, base_col, second_col, booked_col, nbins, title_prefix):
    f, (ax1, ax2) = plt.subplots(1,2)
    plot_dict = {ax1: base_col, ax2: second_col}
    
    for ax, col in plot_dict.items():
        h, bins, patches = ax.hist((data.loc[data[booked_col]==True][col], data.loc[data[booked_col]==False][col]), nbins)

        h_ratio = h[0].astype(float)/(h[1]+h[0]).astype(float)
        h_ratio[np.isnan(h_ratio)]=0

        axa = ax.twinx()
        axa.scatter((bins[0:-1]+bins[1:])/2.0, h_ratio, color='r')
        ax.set_title(title_prefix + ': ' + str(col))
        ax.legend(['booked','not booked'])
        axa.set_ylim([0,1])

    plt.show()


    # evaluate the Logistic Regression model by splitting into train and test sets
def RunlLRModel(X_train, y_train,X_test, y_test,penalty='l1',class_weight=None):

    model2 = LogisticRegression(penalty=penalty,class_weight=class_weight)
    model2.fit(X_train, y_train)
    
    predicted = model2.predict(X_test)

    # generate class probabilities
    probs = model2.predict_proba(X_test)
    df=pd.DataFrame(probs[:,1])
    df = df.join(y_test.reset_index(drop=True))
    df.columns = ['probability','booked']
    
    fpr, tpr, _ = metrics.roc_curve(y_test, probs[:,1])
    df_conf = pd.DataFrame(dict(fpr=fpr, tpr=tpr))

    auc = metrics.roc_auc_score(y_test, probs[:,1])
    return df,df_conf,auc
    
def compare_features_disc_withlift_mul(data, base_col, fea_col_list, fea_name = 'global_features',fig_name = 'test', booked_col = 'booked', nbins=15,test_perc = 0.3, clip_ul = 0.95):
        
    df_auc = pd.DataFrame(columns=['feature_name','auc'])
    
    data = data[[base_col,booked_col]+fea_col_list]

    #Side-by-side feature bucket plots
    if '.png' not in fig_name:
        fig_name = fig_name+'.png'
    print base_col,fea_col_list[0]
    plot_two(data, base_col, fea_col_list[0], booked_col, nbins, title_prefix='Raw:'+fig_name.replace('.png',''))
    for index in range(1,len(fea_col_list),2):
        if (index+2 < len(fea_col_list)):
            plot_two(data, fea_col_list[index+2], fea_col_list[index+1], booked_col, nbins, title_prefix='Raw:'+fig_name.replace('.png',''))
            
        else:
            if(index<len(fea_col_list)-1):
                plot_two(data, fea_col_list[index+1], fea_col_list[index], booked_col, nbins, title_prefix='Raw:'+fig_name.replace('.png',''))
            else:
                plot_two(data, fea_col_list[index-1], fea_col_list[index], booked_col, nbins, title_prefix='Raw:'+fig_name.replace('.png',''))

    X = data[[base_col]+fea_col_list]
    y = data[[booked_col]]

    #Fit logistic regression for each model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=20,stratify=y)
    
    df1, df_conf1,auc1 = RunlLRModel(X_train[[base_col]],y_train,X_test[[base_col]],y_test)
    df1 = df1.rename(columns = {'probability':'probability_'+base_col})

    df2, df_conf2,auc2 = RunlLRModel(X_train[[base_col]+fea_col_list],y_train,X_test[[base_col]+fea_col_list],y_test )
    df2 = df2.rename(columns = {'probability':'probability_'+base_col+'_'+fea_name})

    df2 = df2.drop('booked',axis=1)
    
    df1 = df1.join(df2.reset_index(drop=True))

    #Bucketplot for predicted prob vs actuals
    plot_two(df1,'probability_'+base_col,'probability_'+base_col+'_'+fea_name,booked_col, nbins, title_prefix='Logit:'+fig_name.replace('.png',''))

    #Plot ROC with AUC
    rd1 = np.round(auc1, 3)
    df_auc.loc[0] = [base_col,auc1]
    rd2 = np.round(auc2, 3)
    df_auc.loc[1] = [base_col+'_'+fea_name,auc2]
    plt.plot(df_conf1.fpr,df_conf1.tpr, label=fig_name.replace('.png','')+':'+base_col + ' auc: ' +str(rd1))
    plt.plot(df_conf2.fpr, df_conf2.tpr, label=fig_name.replace('.png','')+':'+base_col+'_'+fea_name+' auc: ' +str(rd2))


    plt.legend(fontsize=20)
    plt.show()
    df_auc = df_auc.sort_values(by = 'auc',ascending=False)
    lift = auc2 - auc1
    df_predicted = pd.concat([df1,df2,y_test])
    
    return df_auc,lift,df_predicted

def compare_features(data, base_col, fea_col, fig_name = '', booked_col = 'booked', nbins=15,test_perc = 0.3, clip_ul = 0.95, clip_comparison=False,fillna=False):
    df_auc = pd.DataFrame(columns=['feature_name','auc'])
    data = data[['conversation_id',base_col,fea_col,booked_col]]
    pct_with_both = (data[(data[base_col].notnull()) & (data[fea_col].notnull())]['conversation_id'].count() + 0.0)/data['conversation_id'].count()

    if (pd.isnull(data[base_col]).count() > 0):
        if(fillna == True):
            data = data.fillna(0)
        else:
            data = data.dropna()
        

    if clip_comparison is True:
        ul = data[fea_col].quantile(clip_ul)
        data[fea_col + '_clipped'] = data[fea_col].clip(0,ul)
        fea_col = fea_col + '_clipped'

    #Side-by-side feature bucket plots
    if '.png' not in fig_name:
        fig_name = fig_name+'.png'

    plot_two(data, base_col, fea_col, booked_col, nbins, title_prefix='Raw:'+fig_name.replace('.png',''))

    X = data[[base_col,fea_col]]
    y = data[[booked_col]]

    #Fit logistic regression for each model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=20,stratify=y)
    
      
    df1, df_conf1,auc1 = RunlLRModel(X_train[[base_col]],y_train,X_test[[base_col]],y_test)
    df1 = df1.rename(columns = {'probability':'probability_'+base_col})

    df2, df_conf2,auc2 = RunlLRModel(X_train[[base_col,fea_col]],y_train,X_test[[base_col,fea_col]],y_test )
    df2 = df2.rename(columns = {'probability':'probability_'+base_col+'_'+fea_col})

    df2 = df2.drop('booked',axis=1)
    
    df1 = df1.join(df2.reset_index(drop=True))

    #Bucketplot for predicted prob vs actuals
    plot_two(df1,'probability_'+base_col,'probability_'+base_col+'_'+fea_col,booked_col, nbins, title_prefix='Logit:'+fig_name.replace('.png',''))

    #Plot ROC with AUC
    rd1 = np.round(auc1, 3)
    df_auc.loc[0] = [base_col,auc1]
    rd2 = np.round(auc2, 3)
    df_auc.loc[1] = [base_col+'_'+fea_col,auc2]
    plt.plot(df_conf1.fpr,df_conf1.tpr, label=fig_name.replace('.png','')+':'+base_col + ' auc: ' +str(rd1))
    plt.plot(df_conf2.fpr, df_conf2.tpr, label=fig_name.replace('.png','')+':'+base_col+'_'+fea_col+' auc: ' +str(rd2))

    plt.legend(fontsize=20)
    plt.show()
    df_auc = df_auc.sort_values(by = 'auc',ascending=False)
    #print("{0:%} of rows have non-null values for both {1} and {2}".format(pct_with_both, base_col, fea_col))
    return pct_with_both,df_auc

def compare_features_disc(data, base_col, fea_col, fig_name = 'test', booked_col = 'booked', nbins=15,test_perc = 0.3, clip_ul = 0.95):
    df_auc = pd.DataFrame(columns=['feature_name','auc'])
    data = data[['conversation_id',base_col,fea_col,booked_col]]
    pct_with_both = data[(data[base_col].notnull()) & (data[fea_col].notnull()) &(data[fea_col]!=0)]['conversation_id'].count()/(data['conversation_id'].count()+0.0)
    

    if (pd.isnull(data[base_col]).count() > 0):
        data = data.dropna()
    #    print 'Nan is dropped, data might be incomplete'

    #Side-by-side feature bucket plots
    if '.png' not in fig_name:
        fig_name = fig_name+'.png'

    plot_two(data, base_col, fea_col, booked_col, nbins, title_prefix='Raw:'+fig_name.replace('.png',''))

    X = data[[base_col,fea_col]]
    y = data[[booked_col]]

    #Fit logistic regression for each model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=20,stratify=y)
    
    df1, df_conf1,auc1 = RunlLRModel(X_train[[base_col]],y_train,X_test[[base_col]],y_test)
    df1 = df1.rename(columns = {'probability':'probability_'+base_col})

    df2, df_conf2,auc2 = RunlLRModel(X_train[[base_col,fea_col]],y_train,X_test[[base_col,fea_col]],y_test )
    df2 = df2.rename(columns = {'probability':'probability_'+base_col+'_'+fea_col})

    df2 = df2.drop('booked',axis=1)
    
    df1 = df1.join(df2.reset_index(drop=True))

    #Bucketplot for predicted prob vs actuals
    plot_two(df1,'probability_'+base_col,'probability_'+base_col+'_'+fea_col,booked_col, nbins, title_prefix='Logit:'+fig_name.replace('.png',''))

    #Plot ROC with AUC
    rd1 = np.round(auc1, 3)
    df_auc.loc[0] = [base_col,auc1]
    rd2 = np.round(auc2, 3)
    df_auc.loc[1] = [base_col+'_'+fea_col,auc2]
    plt.plot(df_conf1.fpr,df_conf1.tpr, label=fig_name.replace('.png','')+':'+base_col + ' auc: ' +str(rd1))
    plt.plot(df_conf2.fpr, df_conf2.tpr, label=fig_name.replace('.png','')+':'+base_col+'_'+fea_col+' auc: ' +str(rd2))


    plt.legend(fontsize=20)
    plt.show()
    df_auc = df_auc.sort_values(by = 'auc',ascending=False)
    #print("{0:%} of rows have non-null values for both {1} and {2}".format(pct_with_both, base_col, fea_col))
    return pct_with_both,df_auc

def compare_features_disc_withlift(data, base_col, fea_col, fig_name = 'test', booked_col = 'booked', nbins=15,test_perc = 0.3, clip_ul = 0.95):
    df_auc = pd.DataFrame(columns=['feature_name','auc'])
    data = data[['conversation_id',base_col,fea_col,booked_col]]
    pct_with_both = data[(data[base_col].notnull()) & (data[fea_col].notnull()) &(data[fea_col]!=0)]['conversation_id'].count()/(data['conversation_id'].count()+0.0)
    

    if (pd.isnull(data[base_col]).count() > 0):
        data = data.dropna()
    #    print 'Nan is dropped, data might be incomplete'

    #Side-by-side feature bucket plots
    if '.png' not in fig_name:
        fig_name = fig_name+'.png'

    plot_two(data, base_col, fea_col, booked_col, nbins, title_prefix='Raw:'+fig_name.replace('.png',''))

    X = data[[base_col,fea_col]]
    y = data[[booked_col]]

    #Fit logistic regression for each model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=20,stratify=y)
    
    df1, df_conf1,auc1 = RunlLRModel(X_train[[base_col]],y_train,X_test[[base_col]],y_test)
    df1 = df1.rename(columns = {'probability':'probability_'+base_col})

    df2, df_conf2,auc2 = RunlLRModel(X_train[[base_col,fea_col]],y_train,X_test[[base_col,fea_col]],y_test )
    df2 = df2.rename(columns = {'probability':'probability_'+base_col+'_'+fea_col})

    df2 = df2.drop('booked',axis=1)
    
    df1 = df1.join(df2.reset_index(drop=True))

    #Bucketplot for predicted prob vs actuals
    plot_two(df1,'probability_'+base_col,'probability_'+base_col+'_'+fea_col,booked_col, nbins, title_prefix='Logit:'+fig_name.replace('.png',''))

    #Plot ROC with AUC
    rd1 = np.round(auc1, 3)
    df_auc.loc[0] = [base_col,auc1]
    rd2 = np.round(auc2, 3)
    df_auc.loc[1] = [base_col+'_'+fea_col,auc2]
    plt.plot(df_conf1.fpr,df_conf1.tpr, label=fig_name.replace('.png','')+':'+base_col + ' auc: ' +str(rd1))
    plt.plot(df_conf2.fpr, df_conf2.tpr, label=fig_name.replace('.png','')+':'+base_col+'_'+fea_col+' auc: ' +str(rd2))


    plt.legend(fontsize=20)
    plt.show()
    df_auc = df_auc.sort_values(by = 'auc',ascending=False)
    lift = auc2 - auc1
    #print("{0:%} of rows have non-null values for both {1} and {2}".format(pct_with_both, base_col, fea_col))
    return df_auc,lift


def compare_features_disc_withlift_mul(data, base_col, fea_col_list, fea_name = 'global_features',fig_name = 'test', booked_col = 'booked', nbins=15,test_perc = 0.3, clip_ul = 0.95):
        
    df_auc = pd.DataFrame(columns=['feature_name','auc'])
    
    data = data[[base_col,booked_col]+fea_col_list]

    #Side-by-side feature bucket plots
    if '.png' not in fig_name:
        fig_name = fig_name+'.png'
    print base_col,fea_col_list[0]
    plot_two(data, base_col, fea_col_list[0], booked_col, nbins, title_prefix='Raw:'+fig_name.replace('.png',''))
    for index in range(1,len(fea_col_list),2):
        if (index+2 < len(fea_col_list)):
            plot_two(data, fea_col_list[index+2], fea_col_list[index+1], booked_col, nbins, title_prefix='Raw:'+fig_name.replace('.png',''))
            
        else:
            if(index<len(fea_col_list)-1):
                plot_two(data, fea_col_list[index+1], fea_col_list[index], booked_col, nbins, title_prefix='Raw:'+fig_name.replace('.png',''))
            else:
                plot_two(data, fea_col_list[index-1], fea_col_list[index], booked_col, nbins, title_prefix='Raw:'+fig_name.replace('.png',''))

    X = data[[base_col]+fea_col_list]
    y = data[[booked_col]]

    #Fit logistic regression for each model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=20,stratify=y)
    
    df1, df_conf1,auc1 = RunlLRModel(X_train[[base_col]],y_train,X_test[[base_col]],y_test)
    df1 = df1.rename(columns = {'probability':'probability_'+base_col})

    df2, df_conf2,auc2 = RunlLRModel(X_train[[base_col]+fea_col_list],y_train,X_test[[base_col]+fea_col_list],y_test )
    df2 = df2.rename(columns = {'probability':'probability_'+base_col+'_'+fea_name})

    df2 = df2.drop('booked',axis=1)
    
    df1 = df1.join(df2.reset_index(drop=True))

    #Bucketplot for predicted prob vs actuals
    plot_two(df1,'probability_'+base_col,'probability_'+base_col+'_'+fea_name,booked_col, nbins, title_prefix='Logit:'+fig_name.replace('.png',''))

    #Plot ROC with AUC
    rd1 = np.round(auc1, 3)
    df_auc.loc[0] = [base_col,auc1]
    rd2 = np.round(auc2, 3)
    df_auc.loc[1] = [base_col+'_'+fea_name,auc2]
    plt.plot(df_conf1.fpr,df_conf1.tpr, label=fig_name.replace('.png','')+':'+base_col + ' auc: ' +str(rd1))
    plt.plot(df_conf2.fpr, df_conf2.tpr, label=fig_name.replace('.png','')+':'+base_col+'_'+fea_name+' auc: ' +str(rd2))


    plt.legend(fontsize=20)
    plt.show()
    df_auc = df_auc.sort_values(by = 'auc',ascending=False)
    lift = auc2 - auc1
    df_predicted = pd.concat([df1,df2,y_test])
    
    return df_auc,lift,df_predicted

