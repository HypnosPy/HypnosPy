# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:08:16 2020

@author: marius
"""
import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
from collections import defaultdict

def get_PA(df,ind, wake_time='07:00', bed_time='23:00'):
    #Extracting daily physical activity data (not adjusted to full days yet)
    #Set time interval
    day = df.between_time(wake_time,bed_time, include_start = True, 
                                            include_end = True)
    #Resampling: daily
    day_rd = day.resample('D', base=7).sum()
    #Defining columns for new df
    subject_cols = ['MVPA_mean','MVPA_std','VigPA_mean','VigPA_std','VigPA_days','ST_mean','ST_std','RMR','age', 'sex','weight','height','BMI','VO2max']
    MET_df = pd.DataFrame(columns=subject_cols)
    #Gets PA averages
    MET_df.loc[ind,'MVPA_mean'] = day_rd['MET_MVPA'].mean()
    MET_df.loc[ind,'MVPA_std'] = day_rd['MET_MVPA'].std()
    MET_df.loc[ind,'VigPA_mean'] = day_rd['MET_VigPA'].mean()
    MET_df.loc[ind,'VigPA_std'] = day_rd['MET_VigPA'].std()
    MET_df.loc[ind,'ST_mean'] = day_rd['MET_Sed'].mean()
    MET_df.loc[ind,'ST_std'] = day_rd['MET_Sed'].std()
    MET_df.loc[ind,'VigPA_days'] = np.count_nonzero(day_rd['MET_VigPA'])
    #Gets other summary data from the original df
    if 'predRMR_Oxford2005' in df.columns:
        MET_df.loc[ind,'RMR']=df['predRMR_Oxford2005'][0]
    MET_df.loc[ind,'age']=df['age'][0]
    MET_df.loc[ind,'sex']=df['sex'][0]
    MET_df.loc[ind,'weight']=df['weight'][0]
    MET_df.loc[ind,'height']=df['height'][0]
    MET_df.loc[ind,'BMI'] = MET_df.loc[ind,'weight'] / MET_df.loc[ind,'height']**2
    if 'P_TR_FITNESS_HighPt_est' in df.columns:
        MET_df.loc[ind,'VO2max'] = df['P_TR_FITNESS_HighPt_est'][0]
    return MET_df,day_rd[['ENMO','MET_Sed','MET_MVPA','MET_VigPA']]

def extract_pa_metrics(d, wake_time='07:00', bed_time='23:00'):
    #Extracts physical activity averages into a dataframe
    pa = defaultdict(dict)
    pa_daily = defaultdict(dict)
    for idx in d.keys():
        pa[idx] = pd.DataFrame()
        #Applies function to get PA data only from wake_time to bed_time
        pa[idx], pa_daily[idx]= get_PA(d[idx],idx,wake_time, bed_time)
    pop_pa = pd.concat(pa.values(),axis=0,ignore_index=False)
    return pa, pa_daily, pop_pa