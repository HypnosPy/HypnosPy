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
import nolds

    
def get_nonlin_params(df,col):
    params = defaultdict(dict)
    params['hurst'] = nolds.hurst_rs(df[col],debug_data=False)
    params['DFA'] = nolds.dfa(df[col],debug_data=False)
    params['sampen'] = nolds.sampen(df[col],debug_data=False)
    #params['lyap1'] = nolds.lyap_r(df[col],debug_data=False)
    return params

def get_nonlinear(self,cols = ['ENMO','mean_hr','hrv_ms'],all_data=True, sleep_wise=True,wake_wise=True):
    nonlin = defaultdict(dict)
    df_1 = self.data
    #df_1['sleep_window'] = df_1['sleep_window'].fillna(method='ffill')
    df_1['sleep_window_0.4'] = df_1['sleep_window_0.4'].fillna(method='bfill')
    for col in cols:
        if sleep_wise==True:
           sleep = df_1[df_1['sleep_window_0.4']==2]
           nonlin[col]['sleep'] = get_nonlin_params(sleep,col)
        if wake_wise==True:
            wake = df_1[df_1['sleep_window_0.4']==1]
            nonlin[col]['wake'] = get_nonlin_params(wake,col)
        if all_data==True:
            nonlin[col]['all'] = get_nonlin_params(df_1,col)
    self.nonlinear = nonlin
    return self