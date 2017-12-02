#group services as overnight and daytime services
#mainly to study the interaction between services
import pandas as pd
def GroupServices(df_in,self_service):
    df_out = df_in
    for agg_col in ['pending_overnight_service_requests','pending_overnight_service_pets','overnight_stay_pets','overnight_stays']:
        df_out[agg_col] = 0
    for agg_col in ['pending_daytime_service_requests','pending_daytime_service_pets','daytime_stay_pets','daytime_stays']:
        df_out[agg_col] = 0    
        
    #group overnight service
    for service_type in ['pending_boarding_requests','pending_boarding_pets', 'pending_traveling_requests', 'pending_traveling_pets','boarding_stays','boarding_stay_pets', 'traveling_stays', 'traveling_stay_pets']:
        if self_service not in service_type:
           # print service_type
            if '_requests' in service_type  and 'pending_' in service_type:
                df_out['pending_overnight_service_requests'] = df_out[service_type] + df_out['pending_overnight_service_requests']
            if '_pets' in service_type and 'pending_' in service_type:
                df_out['pending_overnight_service_pets'] = df_out[service_type] + df_out['pending_overnight_service_pets']
            if '_stay_pets' in service_type:
                df_out['overnight_stay_pets'] = df_out[service_type] +  df_out['overnight_stay_pets'] 
            if '_stays' in service_type:
                df_out['overnight_stays'] = df_out[service_type] + df_out['overnight_stays'] 
            df_out = df_out.drop(service_type, axis = 1)           
                
    for service_type in  ['pending_walking_requests','pending_walking_pets', 'pending_dropin_requests', 'pending_dropin_pets','pending_daycare_requests', 'pending_daycare_pets','walking_stays',
 'walking_stay_pets',
 'dropin_stays',
 'dropin_stay_pets',
 'daycare_stays',
 'daycare_stay_pets']:
        if self_service not in service_type:
            if '_requests' in service_type and 'pending_' in service_type:
                df_out['pending_daytime_service_requests'] = df_out[service_type] + df_out['pending_daytime_service_requests']
            if '_pets'  and 'pending_' in service_type:
                df_out['pending_daytime_service_pets'] = df_out['pending_daytime_service_pets'] + df_out[service_type]
            if '_stay_pets' in service_type:
                df_out['daytime_stay_pets'] = df_out[service_type] +  df_out['daytime_stay_pets']
            if '_stays' in service_type:
                df_out['daytime_stays'] = df_out[service_type] + df_out['daytime_stays']
            df_out = df_out.drop(service_type, axis = 1)
            #print service_type
    return df_out
 
