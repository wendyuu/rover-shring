#generate lower or upper confidence interval of CBR
#feature_df: dataframe that contain the booking/requests and will host cbr
#booking_name,request_name,cbr_name: column names
#low_up: lower or upper confidence interval bound.  default is lower
#alpha: significance level. default is 0.05
#method: normal or beta. Default is beta
#fillna: if fill Nan with 0, default is true

import statsmodels
def GenerateCBRCI(feature_df,booking_name,request_name,cbr_name,low_up=None,alpha=None,method=None,fillna=True):
    if alpha == None:
        alpha = 0.05
    if method == None:
        method = 'beta'
    if low_up == None:
        low_up = 'low'
    bookings = feature_df[booking_name]
    requests = feature_df[request_name]
    if(low_up == 'low'):
        feature_df[cbr_name] = statsmodels.stats.proportion.proportion_confint(bookings,requests,alpha=alpha,method=method)[0]
    if(low_up == 'up'):
        feature_df[cbr_name] = statsmodels.stats.proportion.proportion_confint(bookings,requests,alpha=alpha,method=method)[1]
    if (fillna==True):
        feature_df[cbr_name] = feature_df[cbr_name].fillna(0)   
