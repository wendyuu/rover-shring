#explore different features for conversation booking rate prediction
import pandas as pd
import os
import numpy as np
import analytics_lib
from roverdata.db import DataWarehouse
from roverdata.db import RoverDB

data_warehouse = DataWarehouse()
roverdb = RoverDB()

def get_data(filename='temp.csv', data_dir=os.getcwd(), force_download=False,sql = None,engin='data-warehouse',update_date = ''):
    filename = filename.replace('.csv','_'+update_date+'.csv')
    sql = sql.replace('update_date',update_date)

    if force_download or not os.path.exists(os.path.join(data_dir, filename)):
        print sql
        if(engin == 'data-warehouse'):
            df = data_warehouse.query(sql)
        if(engin == 'roverdb'):
            df = roverdb.query(sql)
        df.to_csv(os.path.join(data_dir, filename), index=False)
        print('downlaoded data')
    else:
        df = pd.read_csv(os.path.join(data_dir, filename))
        print('used cached data')
    return df


nbs_query = '''
select
  cc.id as conversation_id
  , service_type
  , cc.is_repeat_customer
  , requester_was_new_customer
  , ignore_from_search_reason
  , cc.is_premier as convo_is_premier
  , nbs.*
from
  roverdb.conversations_conversation cc
  join roverdb.needs_need nn on
    cc.need_id = nn.id
    and nn.is_imported is false
  left join roverdb.search_nbsvariantrawdata nbs on
    cc.service_id = nbs.service_id
where
  cc.added >= 'update_date'
'''

# the most recent three requests and bookings from Rover.com
recent_booking_sql = '''
with
  base as (
    select
      service_id
      , cc.id as conversation_id
      , has_stay as booked
      , row_number() over(partition by cc.service_id order by cc.added desc, cc.id desc) as recency
      , first_value(cc.id) over(partition by cc.service_id order by cc.added asc rows between unbounded preceding and unbounded following) as first_service_convo
      , case
        when overall is null and has_stay= 1
          then 0.85
        when overall >= 4
          then 1.0
        when overall = 3
          then 0.5
        else 0.0
      end as weight
      ,cc.added
    from
      roverdb.conversations_conversation cc
      left join roverdb.stays_providerrating sp on
        sp.conversation_id = cc.id
      left join roverdb.imports_importedrecord iir on iir.object_id = cc.id and iir.content_type_id = 31
      and iir.source_name ~ 'dogvacay'
    where
      cc.ignore_from_search_reason is null and cc.is_repeat_customer = 0 and iir.object_id is null and cc.is_premier = 0
    group by
      1
      , 2
      , 3
      , 6
      , cc.added
  )
select
  base.*
  , first.weight as first_weight
  , second.weight as second_weight
  , third.weight as third_weight
  , first.booked as first_booking
  , second.booked as second_booking
  , third.booked as third_booking
from
  base
  left join base first on
  base.service_id = first.service_id
  and base.recency = first.recency - 1
left join base second on
second.service_id = first.service_id
and first.recency = second.recency - 1
left join base third on
third.service_id = first.service_id
and second.recency = third.recency - 1
where base.added >= 'update_date'
'''

providercbr_query= '''
with
  target_conversations as (
    select distinct
      id as conversation_id
    from
      roverdb.conversations_conversation
    where
      added >= 'update_date'
      and service_type <> 'on-demand-dog-walking'
  )
  , target_providers as (
    select distinct
      provider_id
    from
      target_conversations tc
      join roverdb.conversations_conversation cc on
        cc.id = tc.conversation_id
  )
  , first_stays as (
    select
      conversation_id
      , added as first_stay_added
    from
      roverdb.stays_stay
  )
  , unfiltered_all_conversations as (
    select
      tc.*
      , cc2.id as past_conversation_id
    from
      target_conversations tc
      join roverdb.conversations_conversation cc on
        cc.id = tc.conversation_id
      left join roverdb.conversations_conversation cc2 on
        (cc2.added < dateadd(day,-3,cc.added) or (cc2.added < cc.added and cc2.has_stay=1))
        and cc2.provider_id = cc.provider_id
        and cc2.is_premier = 0
      left join first_stays fs on
        fs.conversation_id = cc2.id
      where
      cc2.service_type <> 'on-demand-dog-walking' and
        (
          (cc2.ignore_from_search_reason is null)
          or (
            fs.first_stay_added < cc.added
            and cc2.ignore_from_search_reason in (
              'duplicate_conversation'
              , 'repeat_only'
              , 'sitter_unavailable'
              , 'sitter_unavailable_manual'
              , 'pet_not_accepted'
              , 'spay_neuter_status'
              , 'sitter_away'
              , 'requester_deactivated'
              , 'outside_service_radius'
              , 'recurring_billing'
              , 'beyond_calendar_limits'
            )
          )
        )
  )
  , pre_conversations_window1 as (
    select
      uac.*
      , row_number() over(partition by uac.conversation_id, cc.requester_id order by cc.added desc, cc.id desc) as pre_recency
      , cc.requester_id
    from
      unfiltered_all_conversations uac
      join roverdb.conversations_conversation cc on
        cc.id = uac.past_conversation_id
        and cc.is_repeat_customer = 0
  )
  , conversations_window as (
    select
      pcw.conversation_id
      , pcw.requester_id
      , row_number() over(partition by pcw.conversation_id order by cc.added desc, cc.id desc) as recency
    from
      pre_conversations_window1 pcw
      join roverdb.conversations_conversation cc on
        cc.id = pcw.past_conversation_id
    where
      pre_recency = 1
  )
  , raw_stays as (
    select
      s.id as prior_stay_id
      , cc.id as prior_conversation_id
      , cc.service_id
      , cc.provider_id
      , cc.requester_id
      , s.added as prior_stay_added
      , cc.service_type
    from
      roverdb.stays_stay s
      join roverdb.conversations_conversation cc on
        cc.id = s.conversation_id
      join target_providers ts on
        ts.provider_id = cc.provider_id
    where
      s.status <> 'cancelled'
      and cc.is_premier = 0
      and cc.service_type <> 'on-demand-dog-walking'
      and (
        (cc.ignore_from_search_reason is null)
        or (
          cc.ignore_from_search_reason in (
            'duplicate_conversation'
            , 'repeat_only'
            , 'sitter_unavailable'
            , 'sitter_unavailable_manual'
            , 'pet_not_accepted'
            , 'spay_neuter_status'
            , 'sitter_away'
            , 'requester_deactivated'
            , 'outside_service_radius'
            , 'recurring_billing'
            , 'beyond_calendar_limits'
          )
        )
      )
  )
 ,all_recent_reviews as (
select tc.conversation_id
, rs.requester_id
, spr.overall
, rs.service_type
, row_number() over(partition by tc.conversation_id, rs.provider_id, rs.requester_id order by spr.added desc, spr.id desc) as rownum
from
target_conversations tc
join roverdb.conversations_conversation cc on
cc.id = tc.conversation_id
left join raw_stays rs on
rs.prior_stay_added < cc.added
and rs.provider_id = cc.provider_id
left join roverdb.stays_providerrating spr on
spr.conversation_id = rs.prior_conversation_id
and spr.poster_id = cc.requester_id
and spr.added < cc.added
)
,recent_reviews as (
    select conversation_id
    , requester_id
    , overall
    , service_type

from all_recent_reviews
where rownum=1
)
  -- SERVICE_TYPE_CHOICES.walking: [0.98, 0.0, 0.0, 0.10, 0.80, 1.0],
  -- SERVICE_TYPE_CHOICES.day_care: [0.87, 0.0, 0.0, 0.20, 0.42, 1.0],
  --            SERVICE_TYPE_CHOICES.drop_in: [0.94, 0.0, 0.0, 0.25, 0.52, 1.0],
  --            SERVICE_TYPE_CHOICES.boarding: [0.67, 0.0, 0.0, 0.10, 0.53, 1.0],
  --            SERVICE_TYPE_CHOICES.traveling: [0.69, 0.0, 0.0, 0.10, 0.59, 1.0],
  , recent_weights as (
    select
      *
      , case
        when service_type = 'dog-walking'
        and overall is null
          then 0.98
        when service_type = 'dog-walking'
        and overall = 5
          then 1.0
        when service_type = 'dog-walking'
        and overall = 4
          then 0.8
        when service_type = 'dog-walking'
        and overall = 3
          then 0.1
        when service_type = 'doggy-day-care'
        and overall is null
          then 0.87
        when service_type = 'doggy-day-care'
        and overall = 5
          then 1.0
        when service_type = 'doggy-day-care'
        and overall = 4
          then 0.42
        when service_type = 'doggy-day-care'
        and overall = 3
          then 0.2
        when service_type = 'drop-in'
        and overall is null
          then 0.94
        when service_type = 'drop-in'
        and overall = 5
          then 1.0
        when service_type = 'drop-in'
        and overall = 4
          then 0.52
        when service_type = 'drop-in'
        and overall = 3
          then 0.25
        when service_type = 'overnight-boarding'
        and overall is null
          then 0.67
        when service_type = 'overnight-boarding'
        and overall = 5
          then 1.0
        when service_type = 'overnight-boarding'
        and overall = 4
          then 0.53
        when service_type = 'overnight-boarding'
        and overall = 3
          then 0.1
        when service_type = 'overnight-traveling'
        and overall is null
          then 0.69
        when service_type = 'overnight-traveling'
        and overall = 5
          then 1.0
        when service_type = 'overnight-traveling'
        and overall = 4
          then 0.59
        when service_type = 'overnight-traveling'
        and overall = 3
          then 0.1
        else 0.0
      end as weight
    from
      recent_reviews
  )
  , raw_data as (
    select
      cw.conversation_id
      , cw.requester_id
      , cw.recency
      , coalesce(rw.weight, 0.0) as weight
    from
      conversations_window cw
      left join recent_weights rw using (conversation_id, requester_id)
    union all
    select
      tc.conversation_id
      , null as requester_id
      , 0 as recency
      , 0.0 as weight
    from
      target_conversations tc
    where
      tc.conversation_id not in (
        select distinct
          conversation_id
        from
          pre_conversations_window1
      )
  )
select
  conversation_id
  , count(distinct requester_id) as all_requests_provider
  , sum(coalesce(weight, 0.0)) as all_bookings_provider
  , count(
    distinct case
      when recency <= 30
        then requester_id
      else null
    end
  )
  as window_requests_provider
  , sum(
    case
      when recency <= 30
        then coalesce(weight, 0.0)
      else 0.0
    end
  )
  as window_bookings_provider
from
  raw_data
group by
  1;
'''

calendar_sql = '''select
  cc.id as conversation_id
  , sh.accepts_more_than_one_client
  , sh.last_updated_calendar
  , sh.spaces_available
  , cc.added as conversation_added
from
  roverdb.conversations_conversation cc
  join(
    select distinct
      original_conversation_id
      , accepts_more_than_one_client
      , last_updated_calendar
      , spaces_available
      , service_id
    from
      roverdb_historical.service_history
  )
  sh on
    sh.original_conversation_id = cc.id
    and cc.service_id = sh.service_id
where
  cc.added >= 'update_date';
 ;'''

responsiveness_sql = '''
with
  base as (
    select
      c.id as conversation_id
      , c.provider_id
      , c.service_type
      , c.need_id
      , c.added as conversation_added
      , case
        when c.provider_first_response_seconds is null
          then 0
        else 1
      end as provider_response
      , cm.message_added as first_provider_message_added
      , c.provider_first_response_seconds
    from
      roverdb.conversations_conversation c
      left join (
        select
          sender_id
          , conversation_id
          , min(added) as message_added
        from
          roverdb.conversations_message
        group by
          1
          , 2
      )
      cm on
        cm.sender_id = c.provider_id
        and cm.conversation_id = c.id
  )

,sitter_response as (
select
  b1.conversation_id
  , avg(b2.provider_first_response_seconds) as avg_sitter_response_seconds
  , avg(b2.provider_response*1.00) as sitter_response_pct
from
  base b1
  left join base b2 on
    b1.provider_id = b2.provider_id
    and b1.conversation_added > b2.conversation_added
group by
  1
  )

,service_response as (
  select
  b1.conversation_id
  , avg(b2.provider_first_response_seconds) as avg_service_response_seconds
  , avg(b2.provider_response*1.00) as service_response_pct
from
  base b1
  left join base b2 on
    b1.provider_id = b2.provider_id
    and b1.service_type = b2.service_type
    and b1.conversation_added > b2.conversation_added
group by
  1
  )

select
base.conversation_id
,base.provider_response
,base.provider_first_response_seconds
,sitter_response.avg_sitter_response_seconds
,sitter_response.sitter_response_pct
,service_response.avg_service_response_seconds
,service_response.service_response_pct
from
base
left join sitter_response on
  base.conversation_id = sitter_response.conversation_id
left join service_response on
  base.conversation_id = service_response.conversation_id
where base.conversation_added >= 'update_date';
'''
base_training_query = '''
with base as (
select
cc.id as conversation_id
,cc.added as conversation_added
,cc.service_type
,cc.ignore_from_search_reason
,cc.is_premier
,cc.is_repeat_customer
,cc.provider_id
,nn.requester_was_new_customer
,case when nn.booked_stay_count > 0 then 1 else 0 end as need_booked
,cc.has_stay as booked
,ss.added as booked_time
,datediff(day, cc.added, ss.added) as days_to_book
,datediff(day, cc.added, getdate()) as days_old
from
roverdb.conversations_conversation cc
join roverdb.needs_need nn on
	cc.need_id = nn.id
	and nn.is_imported is False
left join roverdb.stays_stay ss on
	cc.id = ss.conversation_id
where
	cc.added >= '2017-07-01'
)

select
base.conversation_id
,base.conversation_added
,base.provider_id
,base.service_type
,base.is_repeat_customer
,base.requester_was_new_customer
,base.ignore_from_search_reason
,base.is_premier as convo_is_premier
,case when days_to_book <= 28 then True else False end as booked
,need_booked
, case when search.conversation_id is not null then True else False end as came_from_search
,person_type as provider_type
from
base
join rover.person_import_type pit on pit.person_id = base.provider_id
left join pganalytics.megatron_searcheventconversation search on
	search.conversation_id = base.conversation_id
where  days_old >= 28
'''

base_feature_query = '''select
  cc.id as conversation_id
  , cc.added as conversation_added
  , cc.service_type
  , cc.ignore_from_search_reason
  , cc.is_premier as convo_is_premier
  , cc.is_repeat_customer
  , cc.provider_id
  , nn.requester_was_new_customer
  , case
    when nn.booked_stay_count > 0
      then 1
    else 0
  end as need_booked
  , cc.has_stay as booked
  , ss.added as booked_time
  , case when search.conversation_id is not null then True else False end as came_from_search
from
  roverdb.conversations_conversation cc
  join roverdb.needs_need nn on
    cc.need_id = nn.id
    and nn.is_imported is false
  left join roverdb.stays_stay ss on
    cc.id = ss.conversation_id
  left join pganalytics.megatron_searcheventconversation search on
  	search.conversation_id = cc.id
where
  cc.added >= 'update_date'
'''

DV_Rover_Request_Count_sql = '''
select
  cc_target.id as conversation_id
  , cc_target.provider_id
  , count(
    distinct case
      when is_imported = 1
      and (merged = 1 or claimed = 1)
        then nn.requester_id
      else null
    end
  )
  as count_unique_merged_requesters_DV
  , count(
    distinct case
      when is_imported = 0
        then nn.requester_id
      else null
    end
  )
  as count_unique_requesters_rover
  ,count(
    distinct case
      when is_imported = 1
      and merged = 0
        then nn.requester_id
      else null
    end
  )
  as count_unique_unmerged_requesters_DV
from
  roverdb.conversations_conversation cc_target
  join roverdb.conversations_conversation cc_history on
    cc_history.provider_id = cc_target.provider_id
    and cc_history.added < cc_target.added
  join roverdb.needs_need nn on
    cc_history.need_id = nn.id
  left join roverdb.imports_personmergedstatus ip on
    ip.person_id = cc_target.provider_id
  left join roverdb.imports_personimportstatus ipi on
    ipi.person_id = cc_target.provider_id
where
  date(cc_target.added) >= 'update_date'
group by
  1
  , 2
'''

def get_feature_from_database(start_date = '2017-07-01',purpose='training',data_dir=os.getcwd(),filename = 'feature_database.csv',force_download = False):

    print 'getting feature data on or after ' + start_date

    if(purpose == 'training'):
        print 'getting dataset for training, conversations that are at least 28 days old'
        base = get_data(filename='database_base_training.csv',data_dir=data_dir,update_date=start_date,sql = base_training_query,force_download=force_download)
    else:
        print 'getting dataset for features including all conversations after ' + start_date
        base = get_data(filename='database_base_feature.csv',data_dir=data_dir,update_date=start_date,sql = base_feature_query,force_download=force_download)

    recent_booking = get_data(filename='database_recent_booking.csv',data_dir=data_dir,update_date=start_date,sql = recent_booking_sql,force_download=force_download)
    for slug in ['first','second','third']:
        recent_booking[slug+'_booking'] = (recent_booking[slug+'_weight'] > 0)
        recent_booking[slug+'_request'] = (pd.isnull(recent_booking[slug+'_weight']) == False)

    providercbr = get_data(filename='database_providercbr.csv',data_dir=data_dir,update_date=start_date,sql = providercbr_query,force_download=force_download)
    providercbr['provider_cbr_all'] = providercbr['all_bookings_provider'].div(providercbr['all_requests_provider'])

    calendar = get_data(filename='database_calendar.csv',data_dir=data_dir,update_date=start_date,sql = calendar_sql)
    calendar['calendar_recency'] = (pd.to_datetime(calendar.conversation_added) - pd.to_datetime(calendar.last_updated_calendar)).dt.days

    responsiveness = get_data(filename='database_responsiveness.csv',data_dir=data_dir,update_date=start_date,sql = responsiveness_sql,force_download=force_download)

    DV_Rover_Request_Count = get_data(filename='database_DV_Rover_Request_Count.csv',data_dir=data_dir,update_date=start_date,sql = DV_Rover_Request_Count_sql,force_download=force_download)

    nbs = get_data(filename='database_nbs.csv',data_dir=data_dir,update_date=start_date,sql = nbs_query,engin='roverdb',force_download=force_download)
    nbs.columns = ['conversation_id', 'service_type', 'is_repeat_customer',
       'requester_was_new_customer', 'ignore_from_search_reason',
       'convo_is_premier', 'id', 'added', 'modified',
       'signup_reason', 'expected_earnings', 'available_all_days',
       'rover_training', 'lessonly_training', 'background_check_taken',
       'has_insurance', 'is_premier', 'num_testimonials', 'relative_price',
       'profile_quality', 'facebook_connected', 'num_pets',
       'flexible_availability', 'spaces_available', 'small_dogs',
       'medium_dogs', 'large_dogs', 'giant_dogs', 'dogs_experience_len',
       'description_len', 'num_images', 'avg_testimonial_len',
       'avg_image_resolution', 'gender', 'years_of_experience', 'donation',
       'app_downloaded', 'verified_by', 'used_rover_to_find_sitter',
       'building_type', 'yard_type', 'inspected_home', 'max_dogs',
       'service_id']
    df = base.merge(nbs,on='conversation_id')
    df = df.merge(providercbr,on='conversation_id')
    df = df.merge(responsiveness, on = 'conversation_id')
    df = df.merge(recent_booking[['conversation_id','first_booking','second_booking','third_booking','first_request','second_request','third_request']], on = 'conversation_id')
    df = df.merge(calendar,on='conversation_id')
    df = df.merge(DV_Rover_Request_Count,on='conversation_id')

    for i in list(df):
        if '_y' in i:
            print i
            df = df.drop(i,axis=1)
        elif '_x' in i:
            print i
            df = df.rename(columns={i:i.replace('_x','')})
    df = df.loc[:,~df.columns.duplicated()]
    df['no_previous_request'] = pd.isnull(df.provider_cbr_all)
    df['rover_imported_ratio'] = df.count_unique_requesters_rover/df.count_unique_merged_requesters_dv
    df.loc[df.rover_imported_ratio>1,'rover_imported_ratio'] = 1
    df['rover_imported_ratio_x_cbr'] = df.provider_cbr_all * df.rover_imported_ratio

    filename = filename.replace('.csv','_'+start_date+'.csv')
    print 'saving to file ' + filename
    df.to_csv(os.path.join(data_dir,filename))

    return df
