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
    #params['hurst'] = nolds.hurst_rs(df[col],debug_data=False)
    params['DFA'] = nolds.dfa(df[col],debug_data=False)
    params['sampen'] = nolds.sampen(df[col],debug_data=False)
    #params['lyap1'] = nolds.lyap_r(df[col],debug_data=False)
    return params

def get_nonlinear(self,cols = ['ENMO','mean_hr','hrv_ms']):
    
    nonlin = defaultdict(dict)
    
    for col in cols:
        column = defaultdict(dict)
        column_sleep = defaultdict(dict)
        column_wake = defaultdict(dict)
        for idx in range(len(self.sleep_windows)):
            params = defaultdict(dict)
            params['DFA'] = nolds.dfa(self.sleep_windows[idx][col],debug_data=False)
            params['SampEn'] = nolds.sampen(self.sleep_windows[idx][col],debug_data=False)
            column_sleep[idx] = params
        for idx in range(len(self.wake_windows)):
            params = defaultdict(dict)
            params['DFA'] = nolds.dfa(self.wake_windows[idx][col],debug_data=False)
            params['SampEn'] = nolds.sampen(self.wake_windows[idx][col],debug_data=False)
            column_wake[idx] = params
        column['sleep'] = column_sleep
        column['wake'] = column_wake
        
        nonlin[col] = column
    self.nonlinear = nonlin
    return self
