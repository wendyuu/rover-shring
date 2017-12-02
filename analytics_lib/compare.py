import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from analytics_lib.Bucket_BR_Plot import Bucket_BR_Plot
from sklearn import metrics 
from IPython.core.pylabtools import figsize
figsize(16,8)

np.seterr(divide='ignore', invalid='ignore')
pd.options.mode.chained_assignment = None  # default='warn'

def plot_two(data, base_col, second_col, booked_col, nbins, title_prefix, suptitle=''):
    f, (ax1, ax2) = plt.subplots(1,2)
    plot_dict = {ax1: base_col, ax2: second_col}

    for ax, col in plot_dict.items():
        h, bins, patches = ax.hist((data.loc[data[booked_col]==True][col], data.loc[data[booked_col]==False][col]), nbins)

        h_ratio = h[0].astype(float)/(h[1]+h[0]).astype(float)
        h_ratio[np.isnan(h_ratio)]=0

        axa = ax.twinx()
        axa.scatter((bins[0:-1]+bins[1:])/2.0, h_ratio, color='r')
        ax.set_title(title_prefix + ': ' + str(col), fontsize=15)
        ax.legend(['booked','not booked'], fontsize=15)
        axa.set_ylim([0,1])
        f.suptitle(suptitle, fontsize=20)

    plt.show()

# evaluate the Logistic Regression model by splitting into train and test sets
def RunlLRModel(X_train, y_train,X_test, y_test):
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)

    predicted = model2.predict(X_test)

    # generate class probabilities
    probs = model2.predict_proba(X_test)
    df=pd.DataFrame(probs[:,1])
    df = df.join(y_test.reset_index(drop=True))
    df.columns = ['probability','booked']

    fpr, tpr, _ = skm.roc_curve(y_test, probs[:,1])
    df_conf = pd.DataFrame(dict(fpr=fpr, tpr=tpr))

    auc = skm.roc_auc_score(y_test, probs[:,1])

    return df,df_conf,auc

def compare_features(data, base_col, fea_col,main_title,fig_name = '', booked_col = 'booked',nbins=15,test_perc = 0.3, clip_ul = 0.95, clip_comparison=False,fillna=False):
    df_auc = pd.DataFrame(columns=['feature_name','auc'])
    data = data[['conversation_id',base_col,fea_col,booked_col]]
    pct_with_both = (data[(data[base_col].notnull()) & (data[fea_col].notnull())]['conversation_id'].count() + 0.0)/data['conversation_id'].count()
    ct = data[(data[base_col].notnull()) & (data[fea_col].notnull())]['conversation_id'].count()

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

    plot_two(data, base_col, fea_col, booked_col, nbins, title_prefix='Raw:'+fig_name.replace('.png',''), suptitle=main_title)

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
    plot_two(df1,'probability_'+base_col,'probability_'+base_col+'_'+fea_col,booked_col, nbins, title_prefix='Logit:'+fig_name.replace('.png',''), suptitle='')

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
    return pct_with_both,df_auc,ct

def compare_features_disc(data, base_col, fea_col, fig_name = 'test', booked_col = 'booked', nbins=15,test_perc = 0.3):
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

def highlight_code(x):
    '''
    highlight the maximum in a Series yellow.
    '''
    if x >= .05:
        return 'background-color: green'
    elif x >= 0:
        return 'background-color: gray'
    else:
        return 'background-color: red'
    
def define_sitter_group(x):
    if x <= 2:
        y = 'black'
    elif x <= 15:
        y = 'green'
    else:
        y = 'red'
    return y

def compare_table(data, base_col, fea_col, sitter_seg=False):
    d = []
    
    def define_service_type_id(x):
        if x == 'overnight-boarding':
            y = 1
        elif x == 'overnight-traveling':
            y = 2
        elif x == 'dog-walking':
            y = 3
        elif x == 'drop-in':
            y = 4
        elif x == 'doggy-day-care':
            y = 5
    return y
    
    data['service_type_id'] = data['service_type'].apply(define_service_type_id)
     
    services = data.sort_values(by='service_type_id')['service_type'].unique()
    sitters = ['black','green','red']
    
    for service in services:
        temp = data[data['service_type']==service]
        result = compare_features(temp, base_col, fea_col, main_title=service)
        
        provider_auc = result[1][result[1]['feature_name']=='cbr_all_provider'].auc.values[0]
        feature_auc = result[1][result[1]['feature_name']!='cbr_all_provider'].auc.values[0]
        group = 'all'
        
        d.append({'service':service, 'sitter_group':group,'pct_pop':result[0],'count':result[2],'auc_provider_cbr':provider_auc, 'auc_feature':feature_auc})
        
        if sitter_seg==True:
            for group in sitters:
                t2 = temp[temp['sitter_group']==group]
                result = compare_features(t2, base_col, fea_col, main_title=service + ': ' + group)

                provider_auc = result[1][result[1]['feature_name']=='cbr_all_provider'].auc.values[0]
                feature_auc = result[1][result[1]['feature_name']!='cbr_all_provider'].auc.values[0]

                d.append({'service':service, 'sitter_group':group,'pct_pop':result[0],'count':result[2],'auc_provider_cbr':provider_auc, 'auc_feature':feature_auc})

    Summary = pd.DataFrame(d)
    Summary = Summary[['service','sitter_group','pct_pop','count','auc_provider_cbr','auc_feature']]
    Summary['delta'] = Summary['auc_feature'] - Summary['auc_provider_cbr']
    
    def highlight_code(x):
        if x >= .005:
            return 'background-color: green'
        elif x >= 0:
            return 'background-color: gray'
        else:
            return 'background-color: red'


    Summary = Summary.style.\
    applymap(highlight_code, subset='delta').\
    format("{:.2%}", subset=['pct_pop','auc_provider_cbr','auc_feature','delta'])

    
    return Summary

def compute_loss(X,y):
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=20,stratify=y)
    
    LR = skl.LogisticRegression()
    LR.fit(X_train, y_train)

    pred = LR.predict(X_test)
    prob = LR.predict_proba(X_test)
    
    p = prob[:,1]
    
    print('Brier score loss: {0}'.format(skm.brier_score_loss(y_test, prob[:,1])))
    print('Log loss: {0}'.format(skm.log_loss(y_test, prob[:,1])))
    del LR
    
def compare_models_complete(data, model_features, base_features, booked_col, bucketplot_features=False, base_title='base prediction', model_title='model prediction'):
    
    if np.any(data[model_features].isnull()) == True:
        print('model features contain null')
        return
    
    elif np.any(data[base_features].isnull()) == True:
        print('base features contain null')
        return
    
    train, test = train_test_split(data)

    def fit_model(train, test, feat, booked_col):
        LR = LogisticRegression(random_state=2017)
        LR.fit(train[feat], np.ravel(train[booked_col]))
        
        intercept = LR.intercept_
        coef = LR.coef_

        prob = LR.predict_proba(test[feat])[:,1]
        auc = skm.roc_auc_score(test[booked_col], prob)

        fpr, tpr, thresholds = skm.roc_curve(test[booked_col], prob)
        brier = skm.brier_score_loss(test[booked_col], prob)
        log = skm.log_loss(test[booked_col], prob)

        return prob, auc, brier, log, tpr, fpr, intercept, coef

    def plot_features(data, feature_list, booked_col, layout='column',bin_no = 15):
        
        if layout=='column':
            figsize(8, 8*len(feature_list))
            f, axes = plt.subplots(len(feature_list),1)
       
        elif layout=='row':
            figsize(8*len(feature_list),8)
            f, axes = plt.subplots(1, len(feature_list))

        for i in range(len(feature_list)):
            ax = axes[i]
            col = feature_list[i]

            h, bins, patches = ax.hist((data.loc[data[booked_col]==True, col], data.loc[data[booked_col]==False, col]), bins=bin_no)
            ax.set_title(col, fontsize=15)
            ax.legend(['booked','not booked'], fontsize=15)


            h_ratio = h[0].astype(float)/(h[1]+h[0]).astype(float)
            h_ratio[np.isnan(h_ratio)]=0

            axa = ax.twinx()
            axa.scatter((bins[0:-1]+bins[1:])/2.0, h_ratio, color='r')
            axa.plot(np.linspace(0,1,bin_no), np.linspace(0,1,bin_no), 'b-')
            axa.set_ylim([0,1])

            i +=1
        plt.show()
        figsize(16,8)


    model_prob, model_auc, model_brier, model_log, model_tpr, model_fpr, model_intercept, model_coef = fit_model(train, test, model_features, booked_col)
    base_prob, base_auc, base_brier, base_log, base_tpr, base_fpr, base_intercept, base_coef = fit_model(train, test, base_features, booked_col)   

    test[model_title] = model_prob
    test[base_title] = base_prob

    #bucketplot features
    
    if bucketplot_features ==True:

        plot_features(data=test, feature_list=model_features, booked_col=booked_col)
    
    plot_features(data=test, feature_list=[model_title,base_title], booked_col=booked_col, layout='row')

    # plot lift
    plt.plot(model_fpr, model_tpr, label=model_title+': '+ str(model_auc))
    plt.plot(base_fpr, base_tpr, label=base_title+': '+str(base_auc))
    plt.legend(fontsize=15)
    plt.show()

    d = pd.DataFrame(columns=['base','model','diff','% diff'], index=['observation_count','auc','brier_score_loss','log_loss'])
    
    d['base'] = [len(data), base_auc, base_brier, base_log]
    d['model'] = [len(data), model_auc, model_brier, model_log]
    d['diff'] = [0, model_auc - base_auc, base_brier - model_brier, base_log - model_log]
    d['% diff']= [0,(model_auc - base_auc)*100.00 / base_auc, (base_brier - model_brier)*100.0 / model_brier, (base_log- model_log)*100.00/ model_log]

    base_model = pd.DataFrame(index=base_features, columns=['coefficient'])
    base_model['coefficient'] = base_coef[0]
    
    print('Base model: ')
    print(base_model)
    print('intercept: ', base_intercept[0])

    new_model = pd.DataFrame(index=model_features, columns=['coefficient'])
    new_model['coefficient'] = model_coef[0]
    
    print()
    print('New model: ')
    print(new_model)
    print('intercept: ', model_intercept[0])
    
    return d

def compare_raw_prediction(df,nbins=15,title='temp',booked_col = 'booked',br_compare_col = 'global_booking_rate', br_base_col = 'service_cbr_window'):
    from sklearn import metrics
    df_auc = pd.DataFrame(columns=['feature_name','auc'])
    
    plot_two(df,br_compare_col,br_base_col,booked_col,nbins, title_prefix='Raw: ',suptitle=title)

    fpr, tpr, _ = metrics.roc_curve(df[booked_col],df[br_compare_col])
    df_global = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    auc_global = metrics.roc_auc_score(df[booked_col].astype(int),df[br_compare_col])

    
    fpr, tpr, _ = metrics.roc_curve(df[[booked_col]],df[[br_base_col]])
    df_cbr = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    auc_cbr = metrics.roc_auc_score(df[booked_col].astype(int),df[br_base_col])
    #Plot ROC with AUC
    rd1 = np.round(auc_global, 3)
    df_auc.loc[0] = [br_compare_col,auc_global]
    rd2 = np.round(auc_cbr, 3)
    df_auc.loc[1] = [br_base_col,auc_cbr]
    plt.plot(df_global.fpr,df_global.tpr, label=title.replace('.png','')+':'+br_compare_col + ' auc: ' +str(rd1))
    plt.plot(df_cbr.fpr, df_cbr.tpr, label=title.replace('.png','')+':'+br_base_col+' auc: ' +str(rd2))

    plt.legend(fontsize=20)
    plt.show()
    df_auc = df_auc.sort_values(by = 'auc',ascending=False)
    lift = auc_global - auc_cbr

    return lift,df_auc
