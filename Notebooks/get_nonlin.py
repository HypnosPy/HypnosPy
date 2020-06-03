# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:22:35 2020

@author: marius
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
from os import path
from collections import defaultdict

def extract_nonlin(d):
    if path.exists('nonlin_pop.csv')==False:
        nonlin,nonlin_pop = get_nonlin_pop(d,cols=['ENMO','mean_hr'],all_data=True)
    else:
        nonlin_pop = pd.read_csv('nonlin_pop.csv')
    #print(nonlin_pop)
    if path.exists('nonlin_pop_day.csv')==False:
        nonlin_day,nonlin_pop_day = get_nonlin_pop(d,cols=['ENMO','mean_hr'], name='day',all_data=False, start_time='08:00', end_time='22:00')
        nonlin_pop_day = nonlin_pop_day.add_suffix('_d')
    else:
        nonlin_pop_day = pd.read_csv('nonlin_pop_day.csv')
        nonlin_pop_day = nonlin_pop_day.add_suffix('_d')
    #print(nonlin_pop_day)
    if path.exists('nonlin_pop_night.csv')==False:    
        nonlin_night,nonlin_pop_night = get_nonlin_pop(d,cols=['ENMO','mean_hr'], name='night',all_data=False, start_time='22:00', end_time='08:00')
        nonlin_pop_night = nonlin_pop_night.add_suffix('_n')
    else:
        nonlin_pop_night = pd.read_csv('nonlin_pop_night.csv')
        nonlin_pop_night = nonlin_pop_night.add_suffix('_n')
    #print(nonlin_pop_night)
    nonlin_pop_all = pd.concat([nonlin_pop,nonlin_pop_day,nonlin_pop_night],axis=1)
    return nonlin_pop_all

def get_nonlin_pop(d,cols,name,all_data=True,start_time='08:00', end_time='22:00'):
    nonlin = defaultdict(dict)
    feature_cols = []
    for col in cols:
        feature_add = ['hurst_'+col,'dfa_'+col,'sampen_'+col,'lyap1_'+col]
        feature_cols.extend(feature_add)
    if all_data==True:
        for col in cols:
            for idx in d.keys():
                df = d[idx].between_time(start_time,end_time, include_start = True, 
                                            include_end = True)    
                nonlin[idx] = get_nonlin_params(df,col,nonlin[idx])
    else:
        for col in cols:
            for idx in d.keys():
                df = d[idx].between_time(start_time,end_time, include_start = True, 
                                            include_end = True)    
                nonlin[idx] = get_nonlin_params(df,col,nonlin[idx])
       
    nonlin_pop = pd.DataFrame(columns = feature_cols)
    for col in feature_cols:
        for idx in nonlin.keys():
            nonlin_pop.loc[idx,col] = nonlin[idx][col]
    
    if all_data==False:
        nonlin_pop.to_csv('nonlin_pop_'+name+'.csv')      

    return nonlin, nonlin_pop    
    
def get_nonlin_params(df,col, nonlin):
    nonlin['hurst_'+col] = nolds.hurst_rs(df[col],debug_data=False)
    nonlin['dfa_'+col] = nolds.dfa(df[col],debug_data=False)
    nonlin['sampen_'+col] = nolds.sampen(df[col],debug_data=False)
    nonlin['lyap1_'+col] = nolds.lyap_r(df[col],debug_data=False)
    #nonlin[idx]['corrdim_'+col],nonlin[idx]['corrdim_data_'+col] = nolds.corr_dim(d[idx][col],emb_dim=1, debug_data=True)
    return nonlin
