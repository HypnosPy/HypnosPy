# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:51:54 2020

@author: marius
"""
#Preprocessing
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta

def get_df_process(filename):
    df = pd.read_csv(filename)
    if 'real_time' in df.columns:
        df["ts"] = [datetime.strptime(ts,"%d-%m-%Y %H:%M:%S") for ts in df["real_time"]] 
    else:    
        df["ts"] = [datetime.strptime(ts,"%d-%b-%Y %H:%M:%S") for ts in df["REALTIME"]] 
    #'2012-09-03"%d-%m-%Y %H:%M:%S"
    df['date'] = df['ts'].dt.date
    df['hour'] = df['ts'].dt.hour
    df.set_index(pd.DatetimeIndex(df['ts']), inplace=True)
    df['ENMO'] = (df["ACC"]/0.0060321) + 0.057
    #Creating some additional columns
    #Sedentary activities counted from 1.5 METs
    df['MET_Sed'] = df['stdMET_highIC_Branch'].apply(lambda x : 1 if x <= 0.5 else 0)
    #WHO recommendations would count moderate PA as between 3-6 METs
    df['MET_MVPA'] = df['stdMET_highIC_Branch'].apply(lambda x : x if x > 2 else 0)
    #WHO recommendations would count moderate PA as >6 METs
    df['MET_VigPA'] = df['stdMET_highIC_Branch'].apply(lambda x : x if x > 5 else 0)
    #Get HRV
    
    file = df
    file = file[file.PWEAR>0] #remove buffer no_wear
    if 'real_time' in file.columns:
        file['real_time'] = pd.to_datetime(file['real_time'], dayfirst=True) #datetime pandas format
    elif 'REAL_TIME' in file.columns:
        file['REAL_TIME'] = pd.to_datetime(file['REAL_TIME'], dayfirst=True) #datetime pandas format
    file = file[file.PWEAR>0] #remove buffer heart rate
    #calculate HRV from IBI (data is corrupted so we should flag/not use that datapoint)
    file['hrv_ms'] = np.where(file.min_ibi_2_in_milliseconds != 1992.0,file['max_ibi_2_in_milliseconds']-file['min_ibi_1_in_milliseconds'], np.nan)
    #file.hrv_milliseconds.fillna(file.hrv_milliseconds.mean(), inplace=True) #fill HRV nans with mean
    file.fillna(file.mean(), inplace=True) #fill nans with mean and if completely empty fill with 0
    file.fillna(0, inplace=True)
    
    df['hrv_ms'] = file['hrv_ms']
    
    return df

def get_data(filelist):
    d = {}
    for idx, file in enumerate(filelist):
        d[idx] = pd.DataFrame()
        d[idx] = get_df_process(file)
    return d
