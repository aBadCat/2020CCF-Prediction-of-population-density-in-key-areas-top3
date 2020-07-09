# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:32:14 2020

@author: dmx
"""


import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def get_feat(data:pd.DataFrame) ->pd.DataFrame:
    shift_feat = []
    
    for i in range(1,8):
        shift_feat.append('Index_history{}'.format(i))
        data['Index_history{}'.format(i)] = data['Index'].shift(len(set(data.Id))*24*i)
    
    b = 1
    c = b+2
    data['mean_Index_history{}_{}'.format(b, c)] = data[['Index_history{}'.format(i) for i in range(b,c)]].mean(axis=1)
    data['max_Index_history{}_{}'.format(b, c)] = data[['Index_history{}'.format(i) for i in range(b,c)]].max(axis=1)
    data['min_Index_history{}_{}'.format(b, c)] = data[['Index_history{}'.format(i) for i in range(b,c)]].min(axis=1)
    data['std_Index_history{}_{}'.format(b, c)] = data[['Index_history{}'.format(i) for i in range(b,c)]].std(axis=1)
    data['sum_Index_history{}_{}'.format(b, c)] = data[['Index_history{}'.format(i) for i in range(b,c)]].sum(axis=1)
    
    b = 1
    c = b+3
    data['mean_Index_history{}_{}'.format(b, c)] = data[['Index_history{}'.format(i) for i in range(b,c)]].mean(axis=1)
    data['max_Index_history{}_{}'.format(b, c)] = data[['Index_history{}'.format(i) for i in range(b,c)]].max(axis=1)
    data['min_Index_history{}_{}'.format(b, c)] = data[['Index_history{}'.format(i) for i in range(b,c)]].min(axis=1)
    data['std_Index_history{}_{}'.format(b, c)] = data[['Index_history{}'.format(i) for i in range(b,c)]].std(axis=1)
    data['sum_Index_history{}_{}'.format(b, c)] = data[['Index_history{}'.format(i) for i in range(b,c)]].sum(axis=1)
    
    data['cafen_Index_history_1_2'] = -(data['Index_history{}'.format(b)].values - data['Index_history2'])
    data['cafen_Index_history_1_3'] = -(data['Index_history{}'.format(b)].values - data['Index_history3'])
    data['cafen_Index_history_1_4'] = -(data['Index_history{}'.format(b)].values - data['Index_history4'])
    
    
    data['incese_Index_history_1_2'] = -(data['Index_history{}'.format(b)].values - data['Index_history2']) / data['Index_history2']
    data['incese_Index_history_1_3'] = -(data['Index_history{}'.format(b)].values - data['Index_history3']) / data['Index_history3']
    data['incese_Index_history_1_4'] = -(data['Index_history{}'.format(b)].values - data['Index_history4']) / data['Index_history4']
    
    data['incese_Index_history_2_3'] = -(data['Index_history2'].values - data['Index_history3']) / data['Index_history3']
    data['incese_Index_history_2_4'] = -(data['Index_history2'].values - data['Index_history4']) / data['Index_history4']

    data['area_avrage'] = data['area'] / data['Index_history7']
    
    data['mean_Index_history1_7'.format(b, c)] = data[['Index_history{}'.format(i) for i in range(1,8)]].mean(axis=1)
    data['area_avrage_1'] = data['area'] / data['mean_Index_history1_7']

    return data


def get_history_feat(label_field:pd.DataFrame, history_field:pd.DataFrame) ->pd.DataFrame:
    
    data_ = history_field.copy()
    data = label_field.copy()
    
    keys = ['Id']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='Id_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='Id_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='Id_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    
    keys = ['area_type_0']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='area_type_0_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='area_type_0_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index'].var().reset_index(name='area_type_0_Index_var')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='area_type_0_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    keys = ['area_type_1']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='area_type_1_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='area_type_1_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index'].std().reset_index(name='area_type_1_Index_std')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='area_type_1_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    
    features = data_.groupby(keys)['area'].mean().reset_index(name='area_type_1_area_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['area'].max().reset_index(name='area_type_1_area_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['area'].std().reset_index(name='area_type_1_area_std')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['area'].sum().reset_index(name='area_type_1_area_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    keys = ['week']
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='week_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    keys = ['hour']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='hour_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='hour_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='hour_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    
    keys = ['is_weekend']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='is_weekend_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='is_weekend_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index'].sum().reset_index(name='is_weekend_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    keys = ['weekday']
    
    
    keys = ['Grid_x', 'Grid_y']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='Grid_xy_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='Grid_xy_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index'].min().reset_index(name='Grid_xy_Index_min')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='Grid_xy_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)

        
    keys = ['Center_x', 'Center_y']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='Center_xy_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='Center_xy_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index'].min().reset_index(name='Center_xy_Index_min')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='Center_xy_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    
    
    keys = ['Id', 'week']
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='Id_week_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    
    keys = ['Id', 'is_weekend']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='Id_is_weekend_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='Id_is_weekend_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index'].min().reset_index(name='Id_is_weekend_Index_min')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='Id_is_weekendk_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    
    keys = ['area_type_1', 'hour']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='area_type_1_hour_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='area_type_1_hour_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index'].min().reset_index(name='area_type_1_hour_Index_min')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='area_type_1_hour_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    keys = ['Id', 'area_type_1', 'hour']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='Id_area_type_1_hour_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='Id_area_type_1_hour_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index'].min().reset_index(name='Id_area_type_1_hour_Index_min')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='Id_area_type_1_hour_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    keys = ['Id', 'area_type_0', 'hour']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='Id_area_type_0_hour_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='Id_area_type_0_hour_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index'].min().reset_index(name='Id_area_type_0_hour_Index_min')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='Id_area_type_0_hour_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    keys = ['hour', 'is_weekend']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='hour_is_weekend_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='hour_is_weekend_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='hour_is_weekendk_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    keys = ['Id', 'hour', 'is_weekend']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='Id_hour_is_weekend_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='Id_hour_is_weekend_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index'].min().reset_index(name='Id_hour_is_weekend_Index_min')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='Id_hour_is_weekendk_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    keys = ['Id', 'hour']
    features = data_.groupby(keys)['Index_history7'].mean().reset_index(name='Id_hour_Index_history7_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index_history7'].max().reset_index(name='Id_hour_Index_history7_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index_history7'].min().reset_index(name='Id_hour_Index_history7_min')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index_history7'].sum().reset_index(name='Id_hour_Index_history7_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    
    
    keys = ['Id', 'weekday']
    
    keys = ['Id', 'week', 'area_type_1']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='Id_week_area_type_1_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='Id_week_area_type_1_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index'].min().reset_index(name='Id_week_area_type_1_Index_min')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='Id_week_area_type_1_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    
    keys = ['Id', 'weekday', 'area_type_1']
    
    keys = ['area_type_0', 'hour']
    features = data_.groupby(keys)['Index'].mean().reset_index(name='area_type_0_hour_Index_mean')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].max().reset_index(name='area_type_0_hour_Index_max')
    data = pd.merge(data, features, how='left', on=keys)
   
    features = data_.groupby(keys)['Index'].min().reset_index(name='area_type_0_hour_Index_min')
    data = pd.merge(data, features, how='left', on=keys)
    
    features = data_.groupby(keys)['Index'].sum().reset_index(name='area_type_0_hour_Index_sum')
    data = pd.merge(data, features, how='left', on=keys)
    
    return data


def model_1():

    result = pd.DataFrame()
    path = r'./data/'
    area_passenger_info = pd.read_csv(path+'train_dataset/area_passenger_info.csv', 
                                      names=['Id', 'area_name', 'area_type', 'Center_x', 
                                             'Center_y', 'Grid_x', 'Grid_y', 'area'])
    
    train = pd.read_csv(path+'train_dataset/area_passenger_index.csv', 
                                      names=['Id', 'Date', 'Index'])
    
    test = pd.read_csv(path+'test_submit_example.csv', names=['Id', 'Date', 'Index'])
    test['Index'] = np.nan
    
    data = pd.concat([train, test],axis=0)
    data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d%H')
    data = data.merge(area_passenger_info, how='left', on='Id')
    data = data.sort_values(by=['date', 'Id'])
    
    for i in range(31, 38):
        
        data['area_type_0'] = data['area_type'].apply(lambda x:x.split(';')[0])
        data['area_type_1'] = data['area_type'].apply(lambda x:x.split(';')[1])
        data['month'] = data['date'].apply(lambda x: x.month)
        data['day'] = data['date'].apply(lambda x: x.day)
        data['hour'] = data['date'].apply(lambda x: x.hour)
        data['week'] = data['date'].apply(lambda x: x.week)
        del data['area_type'], data['area_name']
    
    
    
        data["weekday"] = data['date'].map(lambda x :x.weekday())
        data['is_weekend'] = data["weekday"].map(lambda x :1 if x==5 or x==6 else 0)
    
        le = LabelEncoder()
        le.fit(data['area_type_0'].tolist() + data['area_type_1'].tolist())
        data['area_type_0'] = le.transform(data['area_type_0'])
        data['area_type_1'] = le.transform(data['area_type_1'])
    
        data = get_feat(data)
    
        def fun(x, y):
            if x==1:
                return y-16
            else:
                return y+15
        
        data['dt'] = list(map(lambda x, y:fun(x, y), data['month'], data['day']))
        
        temp = pd.DataFrame()
        for ii in range(22, 40):
            train_label_field1 = data[data["dt"]==ii]
            train_history_field1 = data[data["dt"].between(ii-3,ii-1)]
            data1 = get_history_feat(train_label_field1,train_history_field1)
            temp = pd.concat([temp, data1])
        data = temp
        del temp   
    
    
        data = data.drop(['Center_x', 'Center_y',  'Date', 'month', 'day', 'area'],axis=1 ) #'Grid_x', 'Grid_y',
    
    
    
        cate_feature = ['is_weekend', 'hour', 'weekday', 'area_type_1', 'area_type_0'] #  'Id', 'Date',   
    
        def get_rmse(y_true, y_pred):
            a = np.sqrt(mean_squared_error(y_true, y_pred))
            Score = 1 / (1+a)
            
            return 'Score', Score, True
    
        model = lgb.LGBMRegressor(
                                num_leaves=63,
                                learning_rate=0.1,
                                n_estimators=670,
                                boosting_type='gbdt',
                                max_depth=7,
                                objective='regression',
                                eval_metric = {'mse'},
                                min_child_weight=1e-3,
        #                        subsample_for_bin=10000,
                                feature_fraction = 0.77,
        #                        min_child_samples=7,
                                max_bin=670,
                                min_split_gain=0.1,
                                subsample=0.9,
                                subsample_freq=1,
                                colsample_bytree=0.8,
                                reg_alpha=0.25, 
                                reg_lambda=0.3, 
                                random_state=1367,
        #                        n_jobs=3,
        #                        device = 'gpu',
        #                        gpu_device_id=-1,
        #                        gpu_platform_id=-1
                                )
    
        train_data = data[data["dt"].between(22,i-1)] #22,i-1
        train_lable = train_data['Index']
        del train_data['Index'], train_data['date'], train_data['dt']
        test_data = data[data["dt"]==i]
        del test_data['Index'], test_data['date'], test_data['dt']
        for x in cate_feature:
            train_data[x] = train_data[x].astype('category')
            test_data[x] = test_data[x].astype('category')
        model.fit(train_data, train_lable, categorical_feature=cate_feature)
        a = pd.Series(model.feature_importances_)
        names = pd.Series(test_data.columns)
        feature_importances =  pd.concat([names,a],axis=1)
        predict = model.predict(test_data, num_iteration=model.best_iteration_)
        predict = [0  if i<0 else i for i in predict]
        predict = pd.Series(predict)
        
        
        result = pd.concat([result, predict])
        print(i-15)
        print(predict.mean())
        
        area_passenger_info = pd.read_csv(path+'train_dataset/area_passenger_info.csv', 
                                      names=['Id', 'area_name', 'area_type', 'Center_x', 
                                             'Center_y', 'Grid_x', 'Grid_y', 'area'])
        train = pd.read_csv(path+'train_dataset/area_passenger_index.csv', 
                                      names=['Id', 'Date', 'Index'])
        test = pd.read_csv(path+'test_submit_example.csv', names=['Id', 'Date', 'Index'])
        test['Index'] = np.nan
        data = pd.concat([train, test],axis=0)
        data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d%H')
        data = data.merge(area_passenger_info, how='left', on='Id')
        data = data.sort_values(by=['date', 'Id'])
        data['month'] = data['date'].apply(lambda x: x.month)
        data['day'] = data['date'].apply(lambda x: x.day)
        data['dt'] = list(map(lambda x, y:fun(x, y), data['month'], data['day']))
        data.loc[data["dt"].between(31,i), 'Index'] = result.values
    
    z = result[0:23928*2]
    x = pd.concat([result, z])
    sub = pd.read_csv(path+'test_submit_example.csv', names=['Id', 'Date', 'Index'])
    sub = sub.sort_values(by=['Date', 'Id'])
    sub['Index'] = x.values
    sub = sub.sort_values(by=['Id', 'Date'])
    
    print(sub.Index.mean())
    sub['Index'] = sub['Index']*1.04
    sub.loc[sub["Id"]==786, 'Index'] = sub.loc[sub["Id"]==786, 'Index'].values*1.01
    
    
    sub['Index'] = sub['Index'].apply(lambda x: 0 if x<0 else x)
    print(sub.Index.mean())

    return sub


def model_2():

    result = pd.DataFrame()
    path = r'./data/'
    area_passenger_info = pd.read_csv(path+'train_dataset/area_passenger_info.csv', 
                                      names=['Id', 'area_name', 'area_type', 'Center_x', 
                                             'Center_y', 'Grid_x', 'Grid_y', 'area'])
    
    train = pd.read_csv(path+'train_dataset/area_passenger_index.csv', 
                                      names=['Id', 'Date', 'Index'])
    
    test = pd.read_csv(path+'test_submit_example.csv', names=['Id', 'Date', 'Index'])
    test['Index'] = np.nan
    
    data = pd.concat([train, test],axis=0)
    data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d%H')
    data = data.merge(area_passenger_info, how='left', on='Id')
    data = data.sort_values(by=['date', 'Id'])
    
    for i in range(31, 38):
        
        data['area_type_0'] = data['area_type'].apply(lambda x:x.split(';')[0])
        data['area_type_1'] = data['area_type'].apply(lambda x:x.split(';')[1])
        data['month'] = data['date'].apply(lambda x: x.month)
        data['day'] = data['date'].apply(lambda x: x.day)
        data['hour'] = data['date'].apply(lambda x: x.hour)
        data['week'] = data['date'].apply(lambda x: x.week)
        del data['area_type'], data['area_name']
    
    
    
        data["weekday"] = data['date'].map(lambda x :x.weekday())
        data['is_weekend'] = data["weekday"].map(lambda x :1 if x==5 or x==6 else 0)
    
        le = LabelEncoder()
        le.fit(data['area_type_0'].tolist() + data['area_type_1'].tolist())
        data['area_type_0'] = le.transform(data['area_type_0'])
        data['area_type_1'] = le.transform(data['area_type_1'])
    
        data = get_feat(data)
    
        def fun(x, y):
            if x==1:
                return y-16
            else:
                return y+15
        
        data['dt'] = list(map(lambda x, y:fun(x, y), data['month'], data['day']))
        
        temp = pd.DataFrame()
        for ii in range(22, 40):
            train_label_field1 = data[data["dt"]==ii]
            train_history_field1 = data[data["dt"].between(ii-5,ii-1)]
            data1 = get_history_feat(train_label_field1,train_history_field1)
            temp = pd.concat([temp, data1])
        data = temp
        del temp   
    
    
        data = data.drop(['Center_x', 'Center_y',  'Date', 'month', 'day', 'area'],axis=1 ) #'Grid_x', 'Grid_y',
    
    
    
        cate_feature = ['is_weekend', 'hour', 'weekday', 'area_type_1', 'area_type_0'] #  'Id', 'Date',   
    
        def get_rmse(y_true, y_pred):
            a = np.sqrt(mean_squared_error(y_true, y_pred))
            Score = 1 / (1+a)
            
            return 'Score', Score, True
    
        model = lgb.LGBMRegressor(
                                num_leaves=63,
                                learning_rate=0.1,
                                n_estimators=670,
                                boosting_type='gbdt',
                                max_depth=7,
                                objective='regression',
                                eval_metric = {'mse'},
                                min_child_weight=1e-3,
        #                        subsample_for_bin=10000,
                                feature_fraction = 0.77,
        #                        min_child_samples=7,
                                max_bin=670,
                                min_split_gain=0.1,
                                subsample=0.9,
                                subsample_freq=1,
                                colsample_bytree=0.8,
                                reg_alpha=0.25, 
                                reg_lambda=0.3, 
                                random_state=1367,
        #                        n_jobs=3,
        #                        device = 'gpu',
        #                        gpu_device_id=-1,
        #                        gpu_platform_id=-1
                                )
    
        train_data = data[data["dt"].between(22,i-1)] #22,i-1
        train_lable = train_data['Index']
        del train_data['Index'], train_data['date'], train_data['dt']
        test_data = data[data["dt"]==i]
        del test_data['Index'], test_data['date'], test_data['dt']
        for x in cate_feature:
            train_data[x] = train_data[x].astype('category')
            test_data[x] = test_data[x].astype('category')
        model.fit(train_data, train_lable, categorical_feature=cate_feature)
        a = pd.Series(model.feature_importances_)
        names = pd.Series(test_data.columns)
        feature_importances =  pd.concat([names,a],axis=1)
        predict = model.predict(test_data, num_iteration=model.best_iteration_)
        predict = [0  if i<0 else i for i in predict]
        predict = pd.Series(predict)
        
        
        result = pd.concat([result, predict])
        print(i-15)
        print(predict.mean())
        
        area_passenger_info = pd.read_csv(path+'train_dataset/area_passenger_info.csv', 
                                      names=['Id', 'area_name', 'area_type', 'Center_x', 
                                             'Center_y', 'Grid_x', 'Grid_y', 'area'])
        train = pd.read_csv(path+'train_dataset/area_passenger_index.csv', 
                                      names=['Id', 'Date', 'Index'])
        test = pd.read_csv(path+'test_submit_example.csv', names=['Id', 'Date', 'Index'])
        test['Index'] = np.nan
        data = pd.concat([train, test],axis=0)
        data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d%H')
        data = data.merge(area_passenger_info, how='left', on='Id')
        data = data.sort_values(by=['date', 'Id'])
        data['month'] = data['date'].apply(lambda x: x.month)
        data['day'] = data['date'].apply(lambda x: x.day)
        data['dt'] = list(map(lambda x, y:fun(x, y), data['month'], data['day']))
        data.loc[data["dt"].between(31,i), 'Index'] = result.values
    
    z = result[0:23928*2]
    x = pd.concat([result, z])
    sub = pd.read_csv(path+'test_submit_example.csv', names=['Id', 'Date', 'Index'])
    sub = sub.sort_values(by=['Date', 'Id'])
    sub['Index'] = x.values
    sub = sub.sort_values(by=['Id', 'Date'])
    
    print(sub.Index.mean())
    sub['Index'] = sub['Index']*1.04
    sub.loc[sub["Id"]==786, 'Index'] = sub.loc[sub["Id"]==786, 'Index'].values*1.01
    
    
    sub['Index'] = sub['Index'].apply(lambda x: 0 if x<0 else x)
    print(sub.Index.mean())

    return sub

def get_sub():

    r = model_1()
    y = model_2()
    
    x = r.copy()
    x['Index'] = r['Index'].values*0.6 + y['Index'].values*0.4
    
    def fun1(x,y,z):
        if z>=2020022400 and z<=2020022423:
            return y*1.04
        elif z>=2020022300 and z<=2020022323:
            return y*0.99
        elif z>=2020022100 and z<=2020022123:
            return y*1.03
        elif z>=2020021900 and z<=2020021923:
            return y*0.99
        elif z>=2020021800 and z<=2020021823:
            return y*0.99
        elif z>=2020021700 and z<=2020021723:
            return y*0.99
        elif z>=2020021600 and z<=2020021623: 
            return y*0.95
        else:
            return y
    x['Index'] = list(map(lambda xx,yy,zz:fun1(xx,yy,zz),x['Id'],x['Index'],x["Date"]))
    
    x['Index'] = x['Index'].apply(lambda x: 0 if x<0 else x)
    x = x.sort_values(by=['Id', 'Date']).reset_index().drop(['index'],axis=1)
    
    return x

if __name__ == '__main__':
    dmx = get_sub()
    print(dmx.Index.mean())