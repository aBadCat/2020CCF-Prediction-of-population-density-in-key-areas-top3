    # -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:48:42 2020

@author: lx
"""

import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')
start = time.time()
def model_lx():
    df1 = pd.read_csv("./data/area_passenger_index.csv", header=None, 
                      names=['ID','time','people_index'])
    df2 = pd.read_csv("./data/area_passenger_info.csv",header=None,
                     names=['ID','area_name','area_type','Center_x','Center_y','Grid_x',
                            'Grid_y','area'])
    df3 = pd.read_csv("./data/migration_index.csv",header=None,
                     names=['date','departure_province','departure_city','arrival_province',
                           'arrival_city','index'])
    df4 = pd.read_csv("./data/grid_strength_update.csv", header=None, sep='\t',
                     names=['hour','start_grid_x','start_grid_y','end_grid_x','end_grid_y','Index'])
    sub = pd.read_csv("./data/test_submit_example.csv", header=None,
                     names=['ID','time','people_index'])
    
    df1['time'] = pd.to_datetime(df1['time'],format='%Y%m%d%H')
    df1['date'] = df1['time'].dt.date
    df1['day'] = df1['time'].dt.day.apply(lambda x:x-17 if x>=17 else x+14)
    df1['weekday'] = df1['time'].dt.weekday
    df1['hour'] = df1['time'].dt.hour
    df1 = df1[df1['day']>7]
    
    df_train = df1[['ID','time','people_index','day','weekday','hour','date']]
    df_train = pd.merge(df_train, df2, on=['ID'])
    
    df3['date'] = pd.to_datetime(df3['date'],format='%Y%m%d')
    move_index = {}
    for i in set(df3['date']):
        move_index[i] = df3[(df3['arrival_province'] == '北京市') & (df3['date'] == i)]['index'].sum() +\
        df3[(df3['departure_province'] == '北京市') & (df3['date'] == i)]['index'].sum()
    df_train['move_index'] = df_train['date'].map(move_index)
    
    df_train['Grid_x'] = df_train['Grid_x'].round(decimals=4)
    df_train['Grid_y'] = df_train['Grid_y'].round(decimals=4)
    df4['start_grid_x'] = df4['start_grid_x'].round(decimals=4)
    df4['end_grid_x'] = df4['end_grid_x'].round(decimals=4)
    df4['start_grid_y'] = df4['start_grid_y'].round(decimals=4)
    df4['end_grid_y'] = df4['end_grid_y'].round(decimals=4)
    
    start_index = df4.groupby(['start_grid_x','start_grid_y','hour']).agg({'Index':'sum'})
    end_index = df4.groupby(['end_grid_x','end_grid_y','hour']).agg({'Index':'sum'})
    start_index.rename(columns = {'Index':'hour_in_index'},inplace=True)
    end_index.rename(columns = {'Index':'hour_end_index'},inplace=True)
    df_train['start_grid_x'] = df_train['Grid_x']
    df_train['start_grid_y'] = df_train['Grid_y']
    df_train['end_grid_x'] = df_train['Grid_x']
    df_train['end_grid_y'] = df_train['Grid_y']
    df_train = pd.merge(df_train,start_index,on=['start_grid_x','start_grid_y','hour'],how='left',copy=False)
    df_train = pd.merge(df_train,end_index,on=['end_grid_x','end_grid_y','hour'],how='left',copy=False)
    
    df_train['hour_in_index'] = df_train['hour_in_index'].fillna(0)
    df_train['hour_end_index'] = df_train['hour_end_index'].fillna(0)
    
    feat_columns = ['ID','time','people_index','day','weekday','hour','area','area_type','move_index','hour_in_index','hour_end_index']
    label_columns = ['ID','time','people_index','day','weekday','hour','area','area_type']
    train_label = df_train[(df_train['day']<=29) & (df_train['day']>=23)][label_columns]
    train_history = df_train[(df_train['day']<=22) & (df_train['day']>=16)][feat_columns]
    valid_label = df_train[(df_train['day']<=22) & (df_train['day']>=16)][label_columns]
    valid_history = df_train[(df_train['day']<=15) & (df_train['day']>=9)][feat_columns]
    
    label_columns.remove('people_index')
    
    df_test = sub.copy()
    df_test = pd.merge(df_test, df2, on=['ID'])
    df_test['time'] = pd.to_datetime(df_test['time'], format='%Y%m%d%H')
    df_test['weekday'] = df_test['time'].dt.weekday
    df_test['hour'] = df_test['time'].dt.hour
    df_test['day'] = df_test['time'].dt.day.map(lambda x: x + 14)
    test_label = df_test[(df_test['day']<=38) & (df_test['day']>=30)][label_columns]
    test_history = df_train[(df_train['day']<=29) & (df_train['day']>=23)][feat_columns]
    
    def get_feat(label, history):
        day_ = history['day'].max()
        history_1 = history[(history['day']<=day_)&(history['day']>=day_-2)]
        dict_at_kind = dict({'交通设施;火车站': 1,
                             '交通设施;长途汽车站': 1,
                             '交通设施;飞机场': 2,
                             '医疗;综合医院': 3,
                             '教育培训;高等院校': 4,
                             '旅游景点;公园': 5,
                             '旅游景点;动物园': 5,
                             '旅游景点;文物古迹': 5,
                             '旅游景点;植物园': 5,
                             '旅游景点;水族馆': 5,
                             '旅游景点;游乐园': 5,
                             '旅游景点;风景区': 5,
                             '购物;购物中心': 6,
                             '运动健身;体育场馆': 7})
        label['week_encode'] = label['weekday'].map(dict(history.groupby(['weekday']).mean()['people_index']))
        label['hour_encode'] = label['hour'].map(dict(history_1.groupby(['hour']).mean()['people_index']))
        label['ID_encode'] = label['ID'].map(dict(history_1.groupby(['ID']).mean()['people_index']))
        label['area_type_encode'] = label['area_type'].map(dict(history_1.groupby(['area_type']).mean()['people_index']))
        label['area_type_kind'] = label['area_type'].map(dict_at_kind)
        # past people_index
        t = 1
        for i in range(t):
            tmp_d = history[history['day']==day_-i]
            tmp = tmp_d.groupby(['ID','hour'],as_index=False)['people_index'].agg({'past':'mean'})
            # 最近2天减最远一天
            tmp['pi_2-7'] = history[history['day']==day_-1]['people_index'].values - history[history['day']==day_-6]['people_index'].values
            tmp.rename(columns = {'past':'past'+'_'+str(i)},inplace=True)
            label = pd.merge(label,tmp,on=['ID','hour'],how='left',copy=False)
        
        #ID、hour
        tmp = history_1.groupby(['ID','hour'],as_index=False)['people_index'].agg({'ID_hour_mean':'mean',
                                                                                'ID_hour_min':'min',
                                                                                'ID_hour_max':'max'})
        label = pd.merge(label,tmp,on=['ID','hour'],how='left',copy=False)
        
        # ID、Week
        tmp = history.groupby(['ID','weekday'],as_index=False)['people_index'].agg({'ID_week_sum':'sum',
                                                                                   'ID_week_mean':'mean',
                                                                                   'ID_week_min':'min',
                                                                                   'ID_week_max':'max'})
        label = pd.merge(label,tmp,on=['ID','weekday'],how='left',copy=False)
    
        label['day_encode'] = label['day'].factorize()[0]
        label['is_weekend'] = label['weekday'].map(lambda x: 1 if (x==5) or (x==6) else 0)
        
        # 交叉特征
        label['ID_hour_past_-mean'] = label['past_0'] - label['ID_hour_mean']
        label['ID_hour_day_*mean'] = label['day'] * label['ID_hour_mean']
        label['ID_week_day_*mean'] = label['day'] * label['ID_week_mean']
    #     ''''''
    # # # #    area
        label['pi/area'] = label['ID_encode'] / label['area']
        label['weekpi/area'] = label['week_encode'] / label['area']
        label['hourpi/area'] = label['hour_encode'] / label['area']
        label['areapi/area'] = label['area_type_encode'] / label['area']
        
    
        # the same weekday
        tmp = history.groupby(['ID','hour','weekday'],as_index=False)['people_index'].agg({'last_weekday':'mean'})
        label = pd.merge(label,tmp,on=['ID','hour','weekday'],how='left',copy=False)
        label['ID_hour_past_-mean'] = label['last_weekday'] - label['ID_hour_mean']
        ###
        # move_index
        tmp = history.groupby(['ID','weekday'],as_index=False)['move_index'].agg({'last_move_index':'mean'})
        label = pd.merge(label,tmp,on=['ID','weekday'],how='left',copy=False)
        
        # hour_index
        tmp = history.groupby(['ID','hour','weekday'],as_index=False).agg({'hour_in_index':'mean',
                                                                          'hour_end_index':'mean'})
        label = pd.merge(label,tmp,on=['ID','hour','weekday'],how='left',copy=False)
        label['hour_diff'] = label['hour_end_index'] - label['hour_in_index']
        
        
        label.drop(['day_encode'], axis=1, inplace=True)
        return label
        
    train_target = train_label['people_index']
    train_feat = get_feat(train_label, train_history).drop(['time', 'people_index',
           'area_type'], axis = 1)
    valid_target = valid_label['people_index']
    valid_feat = get_feat(valid_label, valid_history).drop([ 'time','people_index', 
           'area_type'], axis = 1)
    test_feat = get_feat(test_label, test_history).drop(['time',
           'area_type'], axis = 1)
    
    def RMSE(y, pred):
        return 1 / (mean_squared_error(y, pred)**0.5 + 1)
    
    model_lgb = lgb.LGBMRegressor(num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
                                    max_depth=-1, learning_rate=0.05, min_child_samples=10, random_state=2019,
                                    n_estimators=800, subsample=0.9, colsample_bytree=0.7,num_threads= -1)
    
    x_train = train_feat
    y_train = train_target
    x_valid = valid_feat
    y_valid = valid_target
    
    model_lgb.fit(x_train, y_train, categorical_feature=['ID'], verbose=1)
    y_hat = model_lgb.predict(x_valid)
    result = model_lgb.predict(test_feat)
    y_hat[y_hat < 0] = 0
    result[result < 0] = 0
    print(RMSE(y_valid, y_hat))
    print(pd.DataFrame({
            'column': test_feat.columns,
            'importance': model_lgb.feature_importances_,
        }).sort_values(by='importance'))
    print(result.max())
    print(result.mean())
    return result,df_train,df_test

def rule(df_train, df_test):
    # 星期天到星期6
    label_columns = ['ID','time','people_index','day','weekday','hour','area','area_type']
    history_1 = df_train[(df_train['day']<=24) & (df_train['day']>=16)][label_columns]
    history_2 = df_train[(df_train['day']<=29) & (df_train['day']>=22)][label_columns]
    
    label_columns.remove('people_index')
    
    # 星期日
    m = 17
    y_his_1_1 = 0.1*history_1[history_1['day']==m-1]['people_index'].values+ 0.8*history_1[history_1['day']==m]['people_index'].values + 0.1*history_1[history_1['day']==m+1]['people_index'].values
    # 星期一
    m = m + 1
    y_his_1_2 = 0.1*history_1[history_1['day']==m-1]['people_index'].values+ 0.8*history_1[history_1['day']==m]['people_index'].values + 0.1*history_1[history_1['day']==m+1]['people_index'].values
    # 星期二
    m = m + 1
    y_his_1_3 = 0.1*history_1[history_1['day']==m-1]['people_index'].values+ 0.8*history_1[history_1['day']==m]['people_index'].values + 0.1*history_1[history_1['day']==m+1]['people_index'].values
    # 星期三
    m = m + 1
    y_his_1_4 = 0.1*history_1[history_1['day']==m-1]['people_index'].values+ 0.8*history_1[history_1['day']==m]['people_index'].values + 0.1*history_1[history_1['day']==m+1]['people_index'].values
    # 星期四
    m = m + 1
    y_his_1_5 = 0.1*history_1[history_1['day']==m-1]['people_index'].values+ 0.8*history_1[history_1['day']==m]['people_index'].values + 0.1*history_1[history_1['day']==m+1]['people_index'].values
    # 星期五
    m = m + 1
    y_his_1_6 = 0.1*history_1[history_1['day']==m-1]['people_index'].values+ 0.8*history_1[history_1['day']==m]['people_index'].values + 0.1*history_1[history_1['day']==m+1]['people_index'].values
    # 星期六
    m = m + 1
    y_his_1_7 = 0.1*history_1[history_1['day']==m-1]['people_index'].values+ 0.8*history_1[history_1['day']==m]['people_index'].values + 0.1*history_1[history_1['day']==m+1]['people_index'].values
    # 星期日
    m = 23
    y_his_2_1 = 0.1*history_2[history_2['day']==m-1]['people_index'].values+ 0.8*history_2[history_2['day']==m]['people_index'].values + 0.1*history_2[history_2['day']==m+1]['people_index'].values
    # 星期一
    m = m + 1
    y_his_2_2 = 0.1*history_2[history_2['day']==m-1]['people_index'].values+ 0.8*history_2[history_2['day']==m]['people_index'].values + 0.1*history_2[history_2['day']==m+1]['people_index'].values
    # 星期二
    m = m + 1
    y_his_2_3 = 0.1*history_2[history_2['day']==m-1]['people_index'].values+ 0.8*history_2[history_2['day']==m]['people_index'].values + 0.1*history_2[history_2['day']==m+1]['people_index'].values
    # 星期三
    m = m + 1
    y_his_2_4 = 0.1*history_2[history_2['day']==m-1]['people_index'].values+ 0.8*history_2[history_2['day']==m]['people_index'].values + 0.1*history_2[history_2['day']==m+1]['people_index'].values
    # 星期四
    m = m + 1
    y_his_2_5 = 0.1*history_2[history_2['day']==m-1]['people_index'].values+ 0.8*history_2[history_2['day']==m]['people_index'].values + 0.1*history_2[history_2['day']==m+1]['people_index'].values
    # 星期五
    m = m + 1
    y_his_2_6 = 0.1*history_2[history_2['day']==m-1]['people_index'].values+ 0.8*history_2[history_2['day']==m]['people_index'].values + 0.1*history_2[history_2['day']==m+1]['people_index'].values
    # 星期六
    m = m + 1
    y_his_2_7 = 0.1*history_2[history_2['day']==m-1]['people_index'].values+ 0.9*history_2[history_2['day']==m]['people_index'].values
    # 最近
    y_near = 0.5*history_2[history_2['day']==29]['people_index'].values + 0.3*history_2[history_2['day']==28]['people_index'].values + 0.2*history_2[history_2['day']==27]['people_index'].values
    y_trend = (y_his_2_1 + y_his_2_4 + y_his_2_3 + y_his_2_4 + y_his_2_5 + y_his_2_6 + y_his_2_7) /\
              (y_his_1_1 + y_his_1_2 + y_his_1_3 + y_his_1_4 + y_his_1_5 + y_his_1_6 + y_his_1_7)
    y_trend[y_trend.astype(str)=='nan']=0.1
    y_trend[y_trend.astype(str)=='inf']=1.2
    y_trend[(y_trend < 0.7) | (y_trend>1.2)] = (y_trend[(y_trend < 0.7) | (y_trend>1.2)])**0.5
    y_trend_ = [0 for x in range(9)]
    preds = {
        0:1.04,
        1:1.05,
        2:1.06,
        3:1.07,
        4:1.08,
        5:1.09,
        6:1.10,
        7:1.11,
        8:1.12,
    }
    for i in range(9):
        y_trend_[i] = y_trend.copy()
        y_trend_[i][y_trend_[i]<1] = y_trend_[i][y_trend_[i]<1]*preds[i]
        
        y_trend_[i][(y_trend_[i]>1.5)]=1.5
    y_trends = [0 for x in range(9)]
    y_trends[0] = y_his_2_1 / y_his_1_1
    y_trends[1] = y_his_2_2 / y_his_1_2
    y_trends[2] = y_his_2_3 / y_his_1_3
    y_trends[3] = y_his_2_4 / y_his_1_4
    y_trends[4] = y_his_2_5 / y_his_1_5
    y_trends[5] = y_his_2_6 / y_his_1_6
    y_trends[6] = y_his_2_7 / y_his_1_7
    y_trends[7] = y_his_2_1 / y_his_1_1
    y_trends[8] = y_his_2_2 / y_his_1_2
    for i in range(9):
        y_trends[i][y_trends[i].astype(str)=='nan']=0.1
        y_trends[i][y_trends[i].astype(str)=='inf']=1.2
        y_trends[i][(y_trends[i] < 0.7) | (y_trends[i]>1.2)] = (y_trends[i][(y_trends[i] < 0.7) | (y_trends[i]>1.2)])**0.5
        # 140
        y_trends[i][y_trends[i]<1] = y_trends[i][y_trends[i]<1]*preds[i]
        
        y_trends[i][(y_trends[i]>1.5)]=1.5
    result = [0 for x in range(9)]
    r1 = 0.4
    r2 = 0.6
    result[0] = (0.2*y_near + 0.1*y_his_1_1 + 0.7*y_his_2_1)*(y_trend_[0]*r1 + y_trends[0]*r2)
    result[1] = (0.2*y_near + 0.1*y_his_1_2 + 0.7*y_his_2_2)*(y_trend_[1]*r1 + y_trends[1]*r2)
    result[2] = (0.2*y_near + 0.1*y_his_1_3 + 0.7*y_his_2_3)*(y_trend_[2]*r1 + y_trends[2]*r2)
    result[3] = (0.2*y_near + 0.1*y_his_1_4 + 0.7*y_his_2_4)*(y_trend_[3]*r1 + y_trends[3]*r2)
    result[4] = (0.2*y_near + 0.1*y_his_1_5 + 0.7*y_his_2_5)*(y_trend_[4]*r1 + y_trends[4]*r2)
    result[5] = (0.2*y_near + 0.1*y_his_1_6 + 0.7*y_his_2_6)*(y_trend_[5]*r1 + y_trends[5]*r2)
    result[6] = (0.2*y_near + 0.1*y_his_1_7 + 0.7*y_his_2_7)*(y_trend_[6]*r1 + y_trends[6]*r2)
    result[7] = (0.2*y_near + 0.1*y_his_1_1 + 0.7*y_his_2_1)*(y_trend_[7]*r1 + y_trends[7]*r2)
    result[8] = (0.2*y_near + 0.1*y_his_1_2 + 0.7*y_his_2_2)*(y_trend_[8]*r1 + y_trends[8]*r2)
    result_=[]
    for i in range(1, 998):
        for j in range(9):
            result_.extend(result[j][...,24*(i-1):24*i].tolist())
    return np.array(result_)
def get_sub():
    sub = pd.read_csv("./data/test_submit_example.csv", header=None,
                     names=['ID','time','people_index'])
    lx, df_train, df_test = model_lx()
    lx_rule = rule(df_train, df_test)
    sub.iloc[:,2] = lx*0.3 + lx_rule*0.7
    return sub
if __name__ == '__main__':
    
    lx_result = get_sub()
    end = time.time()
    print(end-start)
    print(np.mean(lx_result['people_index']))
    print(np.max(lx_result['people_index']))

