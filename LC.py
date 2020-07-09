# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:59:14 2020

@author: Administrator
"""

import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def get_model_type():   
    model = lgb.LGBMRegressor(
            num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
            max_depth=-1, learning_rate=0.1, min_child_samples=5, random_state=2019,
            n_estimators=600, subsample=0.8, colsample_bytree=0.7)
    return model
def pre(data):
    tmp = data.copy()
    tmp['weekday'] = tmp['Date_Hour'].map(lambda x: datetime.strptime(str(x), '%Y%m%d%H').weekday()+1)
    tmp['month'] = tmp['Date_Hour'].map(lambda x: int((x%2020000000)/10000))
    tmp['day'] = tmp['Date_Hour'].map(lambda x: int((x%10000)/100))
    tmp['hour'] = tmp['Date_Hour'].map(lambda x: int((x%100)))
    return tmp
def get_feature(train, features, cate_feat):
    
    begin_data,last_data= 2020020100, 2020020723
    data = train.copy()

    time_feat = []
    data['is_weekday']=data['weekday'].map(lambda x : 1 if x>=6 else 0)
    time_feat.append('is_weekday')
    data['is_worktime'] = list(map(lambda x,y : 1 if (y<=6) and ( (x>=8 and x<=12) or (x>=14 and x<=18) )  else 0, data['hour'],data['weekday']) )
    time_feat.append('is_worktime')
    data['is_day'] = data['hour'].map(lambda x:1 if x>=8 and x<=19 else 0)
    time_feat.append('is_day')
    data['is_night'] = data['hour'].map(lambda x:1 if x>=21 else 0)
    data['is_eat'] = data['hour'].map(lambda x:1 if ( x>=7 and x<=9 ) or (x>=12 and x<=13) or (x>=17 and x<=18) else 0)

    pivot = pd.pivot_table(data,index=["area_type_2"],values='area',aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'area':'mean_area'}).reset_index()
    data  = pd.merge(data,pivot,on=["area_type_2"],how="left")
    pivot = pd.pivot_table(data,index=["area_type_2"],values='area',aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'area':'min_area'}).reset_index()
    data  = pd.merge(data,pivot,on=["area_type_2"],how="left")   
    
    pivot = pd.pivot_table(data,index=["area_type_2"],values='area',aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={'area':'max_area'}).reset_index()
    data  = pd.merge(data,pivot,on=["area_type_2"],how="left")
    
    features.append('mean_area')
    features.append('max_area')
    features.append('min_area')

    tmp =data[(data['Date_Hour']>=begin_data) & (data['Date_Hour']<=last_data)]
    
    pivot = pd.pivot_table(tmp,index=["ID",'hour'],values='Index',aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'mean_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["ID",'hour'],how="left")

    pivot = pd.pivot_table(tmp,index=["ID",'hour'],values='Index',aggfunc=np.std)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'std_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["ID",'hour'],how="left")
    
    pivot = pd.pivot_table(tmp,index=["ID",'hour'],values='Index',aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'min_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["ID",'hour'],how="left")
    
    pivot = pd.pivot_table(tmp,index=["ID",'hour'],values='Index',aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'max_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["ID",'hour'],how="left")
    
    features.append('std_Index')
    features.append('mean_Index')
    features.append('max_Index')
    features.append('min_Index')
    

    

    
    pivot = pd.pivot_table(tmp,index=["area_type_1",'hour'],values='Index',aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'area_type_1mean_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["area_type_1",'hour'],how="left")

    pivot = pd.pivot_table(tmp,index=["area_type_1",'hour'],values='Index',aggfunc=np.std)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'area_type_1std_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["area_type_1",'hour'],how="left")
    
    pivot = pd.pivot_table(tmp,index=["area_type_1",'hour'],values='Index',aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'area_type_1min_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["area_type_1",'hour'],how="left")
    
    pivot = pd.pivot_table(tmp,index=["area_type_1",'hour'],values='Index',aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'area_type_1max_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["area_type_1",'hour'],how="left")
    
    features.append('area_type_1std_Index')
    features.append('area_type_1mean_Index')
    features.append('area_type_1max_Index')
    features.append('area_type_1min_Index')
 
    pivot = pd.pivot_table(tmp,index=["area_type_2",'hour'],values='Index',aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'area_type_2mean_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["area_type_2",'hour'],how="left")

    pivot = pd.pivot_table(tmp,index=["area_type_2",'hour'],values='Index',aggfunc=np.std)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'area_type_2std_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["area_type_2",'hour'],how="left")
    
    pivot = pd.pivot_table(tmp,index=["area_type_2",'hour'],values='Index',aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'area_type_2min_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["area_type_2",'hour'],how="left")
    
    pivot = pd.pivot_table(tmp,index=["area_type_2",'hour'],values='Index',aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'area_type_2max_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["area_type_2",'hour'],how="left")
    
    features.append('area_type_2std_Index')
    features.append('area_type_2mean_Index')
    features.append('area_type_2max_Index')
    features.append('area_type_2min_Index')
    
    
    
    
    pivot = pd.pivot_table(tmp,index=['hour'],values='Index',aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'hourmean_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=['hour'],how="left")

    pivot = pd.pivot_table(tmp,index=['hour'],values='Index',aggfunc=np.std)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'hourstd_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=['hour'],how="left")
    
    pivot = pd.pivot_table(tmp,index=['hour'],values='Index',aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'hourmin_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=['hour'],how="left")
    
    pivot = pd.pivot_table(tmp,index=['hour'],values='Index',aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'hourmax_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=['hour'],how="left")
    
    features.append('hourstd_Index')
    features.append('hourmean_Index')
    features.append('hourmax_Index')
    features.append('hourmin_Index')


    day=9
    tmp= data.copy()
    
    tmp['Date_Hour'] = list(map(lambda x:x + day*100 ,tmp['Date_Hour']))
    tmp['Date_Hour'] = list(map(lambda x,y: 2020020100+x%100 + (int((x%10000)/100)-32)*100 if int((x%10000)/100)>=32 and y == 1 else x,tmp['Date_Hour'],tmp['month']))
    tmp['Date_Hour']  = tmp['Date_Hour'].map(int)
    lt = list(set(list(data['Date_Hour'])))
    tmp = tmp[tmp['Date_Hour'].isin(lt)][['ID','Date_Hour','Index']]
    tmp = tmp.rename(columns={'Index':'last_9_Index'})
    data = pd.merge(data,tmp,how='left',on=['ID','Date_Hour'])
        
    features.append('last_9_Index')
    
    
    day=14
    tmp= data.copy()
    tmp['Date_Hour'] = list(map(lambda x:x + day*100 ,tmp['Date_Hour']))
    tmp['Date_Hour'] = list(map(lambda x,y: 2020020100+x%100 + (int((x%10000)/100)-32)*100 if int((x%10000)/100)>=32 and y == 1 else x,tmp['Date_Hour'],tmp['month']))
    tmp['Date_Hour']  = tmp['Date_Hour'].map(int)
    lt = list(set(list(data['Date_Hour'])))
    tmp = tmp[tmp['Date_Hour'].isin(lt)][['ID','Date_Hour','Index']]
    tmp = tmp.rename(columns={'Index':'last_1_Index'})
    data = pd.merge(data,tmp,how='left',on=['ID','Date_Hour'])
        

    
    tmp =data[(data['Date_Hour']>=begin_data) & (data['Date_Hour']<=last_data)]
    pivot = pd.pivot_table(tmp,index=["ID",'weekday'],values='Index',aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'weekdaymean_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["ID",'weekday'],how="left")

    pivot = pd.pivot_table(tmp,index=["ID",'weekday'],values='Index',aggfunc=np.std)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'weekdaystd_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["ID",'weekday'],how="left")
    
    pivot = pd.pivot_table(tmp,index=["ID",'weekday'],values='Index',aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'weekdaymin_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["ID",'weekday'],how="left")
    
    pivot = pd.pivot_table(tmp,index=["ID",'weekday'],values='Index',aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'weekdaymax_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["ID",'weekday'],how="left")
    
    features.append('weekdaystd_Index')
    features.append('weekdaymean_Index')
    features.append('weekdaymax_Index')
    features.append('weekdaymin_Index')
    
    '317'
    
    pivot = pd.pivot_table(tmp,index=["hour",'is_weekday'],values='Index',aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'hour_week_mean_index'}).reset_index()
    data  = pd.merge(data,pivot,on=["hour",'is_weekday'],how="left")

    pivot = pd.pivot_table(tmp,index=["hour",'is_weekday'],values='Index',aggfunc=np.std)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'hour_week_std_index'}).reset_index()
    data  = pd.merge(data,pivot,on=["hour",'is_weekday'],how="left")
    
    pivot = pd.pivot_table(tmp,index=["hour",'is_weekday'],values='Index',aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'hour_week_min_index'}).reset_index()
    data  = pd.merge(data,pivot,on=["hour",'is_weekday'],how="left")
    
    pivot = pd.pivot_table(tmp,index=["hour",'is_weekday'],values='Index',aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'hour_week_max_index'}).reset_index()
    data  = pd.merge(data,pivot,on=["hour",'is_weekday'],how="left")
    
    features.append('hour_week_mean_index')
    features.append('hour_week_std_index')
    features.append('hour_week_min_index')
    features.append('hour_week_max_index')
    
    
        
    pivot = pd.pivot_table(tmp,index=['is_weekday'],values='Index',aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'_week_mean_index'}).reset_index()
    data  = pd.merge(data,pivot,on=['is_weekday'],how="left")

    pivot = pd.pivot_table(tmp,index=['is_weekday'],values='Index',aggfunc=np.std)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'_week_std_index'}).reset_index()
    data  = pd.merge(data,pivot,on=['is_weekday'],how="left")
    
    pivot = pd.pivot_table(tmp,index=['is_weekday'],values='Index',aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'_week_min_index'}).reset_index()
    data  = pd.merge(data,pivot,on=['is_weekday'],how="left")
    
    pivot = pd.pivot_table(tmp,index=['is_weekday'],values='Index',aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'_week_max_index'}).reset_index()
    data  = pd.merge(data,pivot,on=['is_weekday'],how="left")
        
    features.append('_week_mean_index')
    features.append('_week_std_index')
    features.append('_week_min_index')
    features.append('_week_max_index')
    
    
    tmp =data[(data['Date_Hour']>=begin_data) & (data['Date_Hour']<=last_data)]
    pivot = pd.pivot_table(tmp,index=["ID",'is_weekday'],values='Index',aggfunc=np.sum)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'is_week_sum_Index'}).reset_index()
    tmp = pd.merge(tmp,pivot,on=['ID','is_weekday'],how='left')
    
    pivot = pd.pivot_table(tmp,index=["ID"],values='Index',aggfunc=np.sum)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'week_sum_Index'}).reset_index()
    tmp = pd.merge(tmp,pivot,on=['ID'],how='left')
    tmp['is_week_ratio_week'] = list(map(lambda x,y:0 if y==0 else x/y,tmp['is_week_sum_Index'],tmp['week_sum_Index']))
    now_tmp = tmp[['ID','is_weekday','is_week_ratio_week']]
    
    now_tmp=now_tmp.drop_duplicates()
    data  = pd.merge(data,now_tmp,on=["ID",'is_weekday'],how="left")
    
    features.append('is_week_ratio_week')
    
    
    tmp =data[(data['Date_Hour']>=begin_data) & (data['Date_Hour']<=last_data)]
    pivot = pd.pivot_table(tmp,index=["ID",'is_weekday','hour'],values='Index',aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'ID_weekdaymean_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["ID",'is_weekday','hour'],how="left")

    pivot = pd.pivot_table(tmp,index=["ID",'is_weekday','hour'],values='Index',aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'ID_weekdaymin_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["ID",'is_weekday','hour'],how="left")
    
    pivot = pd.pivot_table(tmp,index=["ID",'is_weekday','hour'],values='Index',aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={'Index':'ID_weekdaymax_Index'}).reset_index()
    data  = pd.merge(data,pivot,on=["ID",'is_weekday','hour'],how="left")
    
    features = features + time_feat
    cate_feat = cate_feat + time_feat
    
    return data, features, cate_feat



def solve():
    'load data'
    info_data = pd.read_csv('./data/area_passenger_info.csv',header=None)
    index_data = pd.read_csv('./data/area_passenger_index.csv',header=None)
    pred = pd.read_csv('./data/test_submit_example.csv',header=None)
    migration_index  = pd.read_csv('./data/migration_index.csv',header=None)
    
    info_data= info_data.rename(columns={0:'ID',1:'area_name',2:'area_type',3:'Center_x',4:'Center_y',5:'Grid_x',6:'Grid_y',7:'area'})
    index_data= index_data.rename(columns={0:'ID',1:'Date_Hour',2:'Index'})
    pred= pred.rename(columns={0:'ID',1:'Date_Hour',2:'Index'})
    pred['Index'] = np.nan
    
    tot_index = pd.concat([index_data,pred])
    
    
    migration_index= migration_index.rename(columns={0:'Date_Hour',1:'departure_province',2:'departure_city',3:'arrival__province',4:'arrival_city',5:'index'})
    migration_index['out_index']=list(map(lambda x,z:z if x=='北京市' else 0,migration_index['departure_province'],migration_index['index']))
    migration_index['in_index']=list(map(lambda x,z:z if x=='北京市' else 0,migration_index['arrival__province'],migration_index['index']))
    pivot = pd.pivot_table(migration_index,values=['out_index','in_index'],index=['Date_Hour'],aggfunc=np.sum).reset_index()
    pivot['weekday'] = pivot['Date_Hour'].map(lambda x: datetime.strptime(str(x), '%Y%m%d').weekday()+1)
    
    
    tmp = pd.pivot_table(pivot,index=["weekday"],values=['in_index','out_index'],aggfunc=np.mean)
    tmp = pd.DataFrame(tmp).rename(columns={'in_index':'week_mean_in_index','out_index':'week_mean_out_index'}).reset_index()
    
    
    tot_index = pre(tot_index)
    tot_index  = pd.merge(tot_index,tmp,on=["weekday"],how="left")
    data = pd.merge(tot_index,info_data,on=['ID'])
    
    
    
    data['area_type_1']=data['area_type'].map(lambda x: str(x).split(";")[0])
    data['area_type_2']=data['area_type'].map(lambda x: str(x).split(";")[1])
    
    area_type_1 = dict(zip(sorted(list(set(data['area_type_1']))), range(0, len(set(data['area_type_1'])))))
    area_type_2 = dict(zip(sorted(list(set(data['area_type_2']))), range(0, len(set(data['area_type_2'])))))
    
    data['area_type_1']=data['area_type_1'].map(area_type_1)
    data['area_type_2']=data['area_type_2'].map(area_type_2)
    
    data['hour_id'] = data['hour'].map(lambda x:int(x/3))
    
    tot_data = data.copy()
        
    
    features = ['weekday','month','day','hour','area_type_1','area_type_2','Center_x','Center_y','Grid_x','Grid_y','area']
    
    features = features +['week_mean_in_index','week_mean_out_index']
    cate_feat=[]
    
    

    data, features,cate_feat= get_feature(tot_data,features,cate_feat)
    train_data = data[(data['Date_Hour']>=2020020800) & (data['Date_Hour']<=2020021523)]
    train_pred = data[(data['Date_Hour']>=2020021600) & (data['Date_Hour']<=2020022423)]
    model = get_model_type()
    model.fit(train_data[features], train_data['Index'],verbose=1)
    train_pred['pred'] = model.predict(train_pred[features])
    train_pred['pred'] = train_pred['pred'].map(lambda x:0 if x<0 else x) 
    sub = train_pred[['ID','Date_Hour','pred']]
#    sub.to_csv('./result/LC_1.csv',header=None,sep=',',index=False)

    return sub
def get_feature_1(train, features, cate_feat,day):
    
    

    data = train.copy()
    time_feat = []
    data['is_weekday']=data['weekday'].map(lambda x : 1 if x>=6 else 0)
    time_feat.append('is_weekday')
    data['is_worktime'] = list(map(lambda x,y : 1 if (y<=6) and ( (x>=8 and x<=12) or (x>=14 and x<=18) )  else 0, data['hour'],data['weekday']) )
    time_feat.append('is_worktime')
    data['is_day'] = data['hour'].map(lambda x:1 if x>=8 and x<=19 else 0)
    time_feat.append('is_day')
    data['is_night'] = data['hour'].map(lambda x:1 if x>=21 else 0)
    data['is_eat'] = data['hour'].map(lambda x:1 if ( x>=7 and x<=9 ) or (x>=12 and x<=13) or (x>=17 and x<=18) else 0)
    
    


    b = day
    day=9
    tmp= data.copy()
    tmp['Date_Hour'] = list(map(lambda x:x + day*100 ,tmp['Date_Hour']))
    tmp['Date_Hour'] = list(map(lambda x,y: 2020020100+x%100 + (int((x%10000)/100)-32)*100 if int((x%10000)/100)>=32 and y == 1 else x,tmp['Date_Hour'],tmp['month']))
    tmp['Date_Hour']  = tmp['Date_Hour'].map(int)
    lt = list(set(list(data['Date_Hour'])))
    tmp = tmp[tmp['Date_Hour'].isin(lt)][['ID','Date_Hour','Index']]
    tmp = tmp.rename(columns={'Index':'last_9_Index'})
    data = pd.merge(data,tmp,how='left',on=['ID','Date_Hour'])

    features.append('last_9_Index')
    
    day=b
    day=7
    tmp= data.copy()
    tmp['Date_Hour'] = list(map(lambda x:x + day*100 ,tmp['Date_Hour']))
    tmp['Date_Hour'] = list(map(lambda x,y: 2020020100+x%100 + (int((x%10000)/100)-32)*100 if int((x%10000)/100)>=32 and y == 1 else x,tmp['Date_Hour'],tmp['month']))
    tmp['Date_Hour']  = tmp['Date_Hour'].map(int)
    lt = list(set(list(data['Date_Hour'])))
    tmp = tmp[tmp['Date_Hour'].isin(lt)][['ID','Date_Hour','Index']]
    tmp = tmp.rename(columns={'Index':'last_7_Index'})
    data = pd.merge(data,tmp,how='left',on=['ID','Date_Hour'])
        
    features.append('last_7_Index')
    

        
    day=14
    tmp= data.copy()
    tmp['Date_Hour'] = list(map(lambda x:x + day*100 ,tmp['Date_Hour']))
    tmp['Date_Hour'] = list(map(lambda x,y: 2020020100+x%100 + (int((x%10000)/100)-32)*100 if int((x%10000)/100)>=32 and y == 1 else x,tmp['Date_Hour'],tmp['month']))
    tmp['Date_Hour']  = tmp['Date_Hour'].map(int)
    lt = list(set(list(data['Date_Hour'])))
    tmp = tmp[tmp['Date_Hour'].isin(lt)][['ID','Date_Hour','Index']]
    tmp = tmp.rename(columns={'Index':'last_14_Index'})
    data = pd.merge(data,tmp,how='left',on=['ID','Date_Hour'])
        
    features.append('last_14_Index')
    
    
    data['diff'] = data['last_7_Index']-data['last_14_Index']
    features.append('diff')
    
    
    features = features + time_feat
    cate_feat = cate_feat + time_feat
    
    return data, features, cate_feat




def solve2():
    info_data = pd.read_csv('./data/area_passenger_info.csv',header=None)
    index_data = pd.read_csv('./data/area_passenger_index.csv',header=None)
    pred = pd.read_csv('./data/test_submit_example.csv',header=None)
    migration_index  = pd.read_csv('./data/migration_index.csv',header=None)
    
    
    info_data= info_data.rename(columns={0:'ID',1:'area_name',2:'area_type',3:'Center_x',4:'Center_y',5:'Grid_x',6:'Grid_y',7:'area'})
    index_data= index_data.rename(columns={0:'ID',1:'Date_Hour',2:'Index'})
    
    pred= pred.rename(columns={0:'ID',1:'Date_Hour',2:'Index'})
    pred['Index'] = np.nan
    
    tot_index = pd.concat([index_data,pred])
    
    
    
    migration_index= migration_index.rename(columns={0:'Date_Hour',1:'departure_province',2:'departure_city',3:'arrival__province',4:'arrival_city',5:'index'})
    migration_index['out_index']=list(map(lambda x,z:z if x=='北京市' else 0,migration_index['departure_province'],migration_index['index']))
    migration_index['in_index']=list(map(lambda x,z:z if x=='北京市' else 0,migration_index['arrival__province'],migration_index['index']))
    pivot = pd.pivot_table(migration_index,values=['out_index','in_index'],index=['Date_Hour'],aggfunc=np.sum).reset_index()
    pivot['weekday'] = pivot['Date_Hour'].map(lambda x: datetime.strptime(str(x), '%Y%m%d').weekday()+1)
    
    
    tmp = pd.pivot_table(pivot,index=["weekday"],values=['in_index','out_index'],aggfunc=np.mean)
    tmp = pd.DataFrame(tmp).rename(columns={'in_index':'week_mean_in_index','out_index':'week_mean_out_index'}).reset_index()
    
    
    tot_index = pre(tot_index)
    
    tot_index  = pd.merge(tot_index,tmp,on=["weekday"],how="left")
    
    data = pd.merge(tot_index,info_data,on=['ID'])
    
    data['area_type_1']=data['area_type'].map(lambda x: str(x).split(";")[0])
    data['area_type_2']=data['area_type'].map(lambda x: str(x).split(";")[1])
    
    area_type_1 = dict(zip(sorted(list(set(data['area_type_1']))), range(0, len(set(data['area_type_1'])))))
    area_type_2 = dict(zip(sorted(list(set(data['area_type_2']))), range(0, len(set(data['area_type_2'])))))
    
    data['area_type_1']=data['area_type_1'].map(area_type_1)
    data['area_type_2']=data['area_type_2'].map(area_type_2)
    
    
    use_data = data.copy()
    

    
    out = data[(data['Date_Hour']>=2020021600) & (data['Date_Hour']<=2020022423)][['ID','Date_Hour','Index']]
    valid = data[(data['Date_Hour']>=2020020700) & (data['Date_Hour']<=2020021523)]
    valid['pred']=-1
    
    for day in range(1,10):
        
        features = ['weekday','month','day','hour','area_type_1','area_type_2','Center_x','Center_y','Grid_x','Grid_y','area']
        features = features +['week_mean_in_index','week_mean_out_index']
        cate_feat =['weekday','month','day','hour','area_type_1','area_type_2'] 
        data, features,cate_feat= get_feature_1(use_data, features, cate_feat, day)
        train_data = data[(data['Date_Hour']>=2020020800) & (data['Date_Hour']<=2020021523)]
        train_pred = data[(data['Date_Hour']>=(20200215+day)*100) & (data['Date_Hour']<=(20200215+day)*100+23)]
        
        model = get_model_type()
        model.fit(train_data[features], train_data['Index'],verbose=1)
        train_pred['pred'] = model.predict(train_pred[features])
        train_pred['pred'] = train_pred['pred'].map(lambda x:0 if x<0 else x) 
    
        out.loc[(out['Date_Hour']>=(20200215+day)*100) & (out['Date_Hour']<=(20200215+day)*100+23),  'Index'] = train_pred['pred'].values
        use_data.loc[(use_data['Date_Hour']>=(20200215+day)*100) & (use_data['Date_Hour']<=(20200215+day)*100+23),  'Index'] = train_pred['pred'].values
        
    sub = out[['ID','Date_Hour','Index']]
    
#    sub.to_csv('./result/LC_2.csv',header=None,sep=',',index=False)
    return sub


def solve3():
    sub_1=solve()
    sub_2=solve2()
    sub_1['pred']= list(map(lambda x,y:x*1.04 if y ==786 else x*1.05,sub_1['pred'] ,sub_1['ID'] ))
    sub_2['Index']= list(map(lambda x,y:x*1.04 if y ==786 else x*1.03,sub_2['Index'] ,sub_2['ID'] ))
    
#    sub_1.to_csv('./result/LC_3.csv',header=None,sep=',',index=False)
#    sub_2.to_csv('./result/LC_4.csv',header=None,sep=',',index=False)
   
    sub_1['Index'] = sub_2['Index']
    sub_1['Pred'] = sub_1['Index']*0.50 + sub_1['pred']*0.50
    out = sub_1[['ID','Date_Hour','Pred']]
    out.loc[(out['Date_Hour']>=2020021600)&(out['Date_Hour']<=2020021623),'Pred'] = out.loc[(out['Date_Hour']>=2020021600)&(out['Date_Hour']<=2020021623),'Pred']*0.95
    out.loc[(out['Date_Hour']>=2020022100)&(out['Date_Hour']<=2020022123),'Pred'] = out.loc[(out['Date_Hour']>=2020022100)&(out['Date_Hour']<=2020022123),'Pred']*1.02
    out.loc[(out['Date_Hour']>=2020022400)&(out['Date_Hour']<=2020022423),'Pred'] = out.loc[(out['Date_Hour']>=2020022400)&(out['Date_Hour']<=2020022423),'Pred']*1.05
    out.to_csv('./result/LC.csv',header=None,sep=',',index=False)

solve3()
