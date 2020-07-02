# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:30:10 2020

@author: marius
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, date, time, timedelta
import matplotlib.pyplot as plt
from os import path
from collections import defaultdict


# Function to compute the sleep windows for each subject, computed using the simple HR heuristic
# takes arguments for the start and finish times of the sleep analysis (at the moment does not look at day naps)
# takes the length of the shortest sequence deemed as sleep and the quantile for the sleep threshold


def get_sleep(self, q_sleep=0.4, bed_time = '20:00',wake_time = '12:00', seq_length = 15):
    if all(self.data.ts.diff()[1:] == np.timedelta64(60, 's')) == False:
        df_all = self.data.resample('1T').mean()
    else:
        df_all = self.data
    
    night = df_all.between_time(bed_time,wake_time, include_start = True, 
                                            include_end = True)
    day_start = '09:00'
    day_end = '22:00'
    col_sleep = 'sleep'
    col_wake = 'wake'
    day = df_all.between_time(day_start,day_end, include_start = True, 
                                            include_end = True)
    #Base is read from function input and start of day
    base_s = int(bed_time[0:2])
    base_w = int(day_start[0:2])
    
    night_count = night['mean_hr'].resample('24H',base=base_s).count()
    
    def get_threshold_bins(df, df2, col_apply, col_new, base_s, base_w,quantile,  sleep_t):
        if sleep_t:
            df[col_new] = df[col_apply].resample('24H',base=base_s).apply(lambda x: np.quantile(x,quantile))
        else:
            df[col_new] = df2[col_apply].resample('24H',base=base_w).apply(lambda x: np.quantile(x,quantile))
        df[col_new] = df[col_new].ffill(axis=0)
        #Incomplete days at the beginning are backfilled with the value for the following day
        df[col_new] = df[col_new].fillna(method='bfill')
        df[col_new+'_bin'] =np.where((df[col_apply]-df[col_new])>0, 0, 1)
        df[col_new+'_bin'] = df[col_new+'_bin'].rolling(window=5).median().fillna(method='bfill')
        df[col_new+'_seq_id'] = df.groupby((df[col_new+'_bin'] != df[col_new+'_bin'].shift(1).fillna(False)).cumsum()).ngroup()
        df[col_new+'_len'] = 0
        df[col_new+'_len'] = night[[col_new+'_len', col_new+'_seq_id']].groupby(col_new+'_seq_id').transform('count')
        return df
    
    def get_sleep_windows(df, data,col_sleep, base_s, min_window_length):
        df[col_sleep+'_labels'] = (df[col_sleep+'_bin']==1.0) & (df[col_sleep+'_len'] > min_window_length)
        df[col_sleep+'_labels'] = (df[col_sleep+'_labels'] == True).astype(int)
        #Extract the times when labelled sleep changes to wake and viceversa
        state_changes = df[col_sleep+'_labels'].diff().fillna(0)[lambda x: x != 0].index.tolist()
        state_changes = pd.to_datetime(state_changes).to_frame()
        #Extract index of nights available for the subject to pass onto sleep df
        index = state_changes.resample('24H',base=base_s).min().index
        sleep_df = pd.DataFrame(columns=['TST','sleep_onset','sleep_offset','weekday'], index = index)
        #Extract sleep onset each night as earliest wake-sleep transition after base hour
        sleep_df['sleep_onset'] = state_changes.resample('24H',base=base_s).min()
        #Extract sleep onset each night as last sleep-wake transition after base hour
        sleep_df['sleep_offset'] = state_changes.resample('24H',base=base_s).max()
        sleep_df['TST']= sleep_df['sleep_offset'] -  sleep_df['sleep_onset']
        sleep_df['weekday'] = sleep_df.index.dayofweek
        sleep_df = pd.DataFrame(sleep_df)
        #Label sleep windows on original df according to sleep_df
        df[col_sleep+'_window'] = np.nan
        df[col_sleep+'_window'].loc[sleep_df['sleep_onset']] = 1
        df[col_sleep+'_window'].loc[sleep_df['sleep_offset']] = 2
        df[col_sleep+'_window'] = df[col_sleep+'_window'].fillna(method='ffill')
        data[col_sleep+'_window'] = df[col_sleep+'_window']
        data[col_sleep+'_window'] = data[col_sleep+'_window'].fillna(method='ffill')
        return df, state_changes, sleep_df, index, data

    def get_wake_windows(df,data,col_wake, col_sleep,base_s):
        #min_len = 5
        max_len = 60
        min_interwake = 10
        df[col_wake+'_window'] = (df[col_wake+'_bin']==0.0) & (df[col_wake+'_len'] < max_len) & (df[col_wake+'_len'] >= 5) & (df[col_sleep+'_window']==1)    
        df[col_wake+'_window'] = (df[col_wake+'_window'] == True).astype(int)
        #Extract the times when labelled wake changes to sleep and viceversa, according to higher HR quantile
        wake_changes = df[col_wake+'_window'].diff().fillna(0)[lambda x: x != 0].index.tolist()
        wake_changes = pd.to_datetime(wake_changes)
        wake_frame = wake_changes.to_frame()
        #Delete state changes where there is less than min_interwake mins of sleep between detected awakenings
        short_wakes = []
        for i in np.arange(2, len(wake_changes)-2,2):
            if (wake_changes[i] - wake_changes[i-1])<=timedelta(minutes=min_interwake):
                short_wakes.append(i-1)
                short_wakes.append(i)
        wake_changes = pd.to_datetime([wake_changes[j] for j in set(range(len(wake_changes))) - set(short_wakes)])
        #Separate onsets and offsets of awakenings
        wake_on = [] 
        wake_off = [] 
        for i in range(len(wake_changes)): 
            if i % 2: 
                wake_off.append(wake_changes[i]) 
            else : 
                wake_on.append(wake_changes[i]) 
        #Extract awakening metrics     
        wake_df = pd.DataFrame(columns=['WASO','AwaNo'],index=index)
        df[col_wake+'_window'] = np.nan
        df[col_wake+'_window'].loc[wake_on] = 1
        wake_df['AwaNo'] = df[col_wake+'_window'].resample('24H',base=base_s).count()
        df[col_wake+'_window'].loc[wake_off] = 0
        df[col_wake+'_window'] = df[col_wake+'_window'].fillna(method='ffill')
        #Extract Wake After Sleep Onsets (minutes), adding numer of awakenings to correct for the missing min at end
        wake_df['WASO'] = df[col_wake+'_window'].resample('24H',base=base_s).sum() + wake_df['AwaNo']
        data[col_wake+'_window'] = df[col_wake+'_window']
        data[col_wake+'_window'] = data[col_wake+'_window'].fillna(method='ffill')
        return df, wake_changes, wake_df,data
    
    #Execute functions
    get_threshold_bins(night, day,'mean_hr',col_sleep , base_s, base_w, quantile=q_sleep ,sleep_t=True)
    get_threshold_bins(night, day,'mean_hr',col_wake, base_s, base_w, quantile=q_sleep,sleep_t=False)
    
    night, state_changes, sleep_df, index,df = get_sleep_windows(night, df_all,col_sleep, base_s,seq_length)
    night, wake_changes, wake_df,df = get_wake_windows(night,df_all,col_wake,col_sleep,base_s)
    
    sleep_df = pd.concat([sleep_df,wake_df], axis=1)
    #Extract sleep efficiency as (TST-WASO)/TST
    sleep_TST_delta = [(x.seconds/60) for x in sleep_df['TST'] ] 
    sleep_df['SEff'] = (sleep_TST_delta - sleep_df['WASO']) / sleep_TST_delta
    #Keep only nights with enough data (more than 10 hours available in the interval)
    sleep_df = sleep_df[night_count > 600]
            
    self.data['sleep_window_'+str(q_sleep)] = df_all['sleep_window']
    self.data['wake_window_'+str(q_sleep)] = df_all['wake_window']
    
    self.sleep_rec = sleep_df
            
    return self


#Function to get the sleep regularity index
def get_SRI(self, q_sleep = 0.4):
    sleep_col = 'sleep_window_'+str(q_sleep)
    sri_delta = np.zeros(len(self.data[self.data.index[0]:self.data.shift(periods=-1,freq='D').index[-1]]))
    for i in range(len(self.data[self.data.index[0]:self.data.shift(periods=-1,freq='D').index[-1]])):
        if self.data[sleep_col][self.data.index[i]] == self.data[sleep_col].shift(periods=-1,freq='D')[self.data.index[i]]:
            sri_delta[i] = 1
        else:
            sri_delta[i] = 0
    sri_df = pd.DataFrame(sri_delta)
    sri = -100 + (200 / (len(self.data[self.data.index[0]:self.data.shift(periods=-1,freq='D').index[-1]]))) * sri_df.sum()
    self.SRI = float(sri)
    return self     

def get_sleep_grid(self, q_sleep=[0.35,0.4], bed_time = '20:00',wake_time = '12:00', seq_length = 15):
    if all(self.data.ts.diff()[1:] == np.timedelta64(60, 's')) == False:
        df_all = self.data.resample('1T').mean()
    else:
        df_all = self.data
    
    night = df_all.between_time(bed_time,wake_time, include_start = True, 
                                            include_end = True)
    day_start = '09:00'
    day_end = '22:00'
    col_sleep = 'sleep'
    col_wake = 'wake'
    day = df_all.between_time(day_start,day_end, include_start = True, 
                                            include_end = True)
    #Base is read from function input and start of day
    base_s = int(bed_time[0:2])
    base_w = int(day_start[0:2])
    
    night_count = night['mean_hr'].resample('24H',base=base_s).count()
    
    def get_threshold_bins(df, df2, col_apply, col_new, base_s, base_w,quantile,  sleep_t):
        if sleep_t:
            df[col_new] = df[col_apply].resample('24H',base=base_s).apply(lambda x: np.quantile(x,quantile))
        else:
            df[col_new] = df2[col_apply].resample('24H',base=base_w).apply(lambda x: np.quantile(x,quantile))
        df[col_new] = df[col_new].ffill(axis=0)
        #Incomplete days at the beginning are backfilled with the value for the following day
        df[col_new] = df[col_new].fillna(method='bfill')
        df[col_new+'_bin'] =np.where((df[col_apply]-df[col_new])>0, 0, 1)
        df[col_new+'_bin'] = df[col_new+'_bin'].rolling(window=5).median().fillna(method='bfill')
        df[col_new+'_seq_id'] = df.groupby((df[col_new+'_bin'] != df[col_new+'_bin'].shift(1).fillna(False)).cumsum()).ngroup()
        df[col_new+'_len'] = 0
        df[col_new+'_len'] = night[[col_new+'_len', col_new+'_seq_id']].groupby(col_new+'_seq_id').transform('count')
        return df
    
    def get_sleep_windows(df, data,col_sleep, base_s, min_window_length):
        df[col_sleep+'_labels'] = (df[col_sleep+'_bin']==1.0) & (df[col_sleep+'_len'] > min_window_length)
        df[col_sleep+'_labels'] = (df[col_sleep+'_labels'] == True).astype(int)
        #Extract the times when labelled sleep changes to wake and viceversa
        state_changes = df[col_sleep+'_labels'].diff().fillna(0)[lambda x: x != 0].index.tolist()
        state_changes = pd.to_datetime(state_changes).to_frame()
        #Extract index of nights available for the subject to pass onto sleep df
        index = state_changes.resample('24H',base=base_s).min().index
        sleep_df = pd.DataFrame(columns=['TST','sleep_onset','sleep_offset','weekday'], index = index)
        #Extract sleep onset each night as earliest wake-sleep transition after base hour
        sleep_df['sleep_onset'] = state_changes.resample('24H',base=base_s).min()
        #Extract sleep onset each night as last sleep-wake transition after base hour
        sleep_df['sleep_offset'] = state_changes.resample('24H',base=base_s).max()
        sleep_df['TST']= sleep_df['sleep_offset'] -  sleep_df['sleep_onset']
        sleep_df['weekday'] = sleep_df.index.dayofweek
        sleep_df = pd.DataFrame(sleep_df)
        #Label sleep windows on original df according to sleep_df
        df[col_sleep+'_window'] = np.nan
        df[col_sleep+'_window'].loc[sleep_df['sleep_onset']] = 1
        df[col_sleep+'_window'].loc[sleep_df['sleep_offset']] = 2
        df[col_sleep+'_window'] = df[col_sleep+'_window'].fillna(method='ffill')
        data[col_sleep+'_window'] = df[col_sleep+'_window']
        data[col_sleep+'_window'] = data[col_sleep+'_window'].fillna(method='ffill')
        return df, state_changes, sleep_df, index, data

    def get_wake_windows(df,data,col_wake, col_sleep,base_s):
        #min_len = 5
        max_len = 60
        min_interwake = 10
        df[col_wake+'_window'] = (df[col_wake+'_bin']==0.0) & (df[col_wake+'_len'] < max_len) & (df[col_wake+'_len'] >= 5) & (df[col_sleep+'_window']==1)    
        df[col_wake+'_window'] = (df[col_wake+'_window'] == True).astype(int)
        #Extract the times when labelled wake changes to sleep and viceversa, according to higher HR quantile
        wake_changes = df[col_wake+'_window'].diff().fillna(0)[lambda x: x != 0].index.tolist()
        wake_changes = pd.to_datetime(wake_changes)
        wake_frame = wake_changes.to_frame()
        #Delete state changes where there is less than min_interwake mins of sleep between detected awakenings
        short_wakes = []
        for i in np.arange(2, len(wake_changes)-2,2):
            if (wake_changes[i] - wake_changes[i-1])<=timedelta(minutes=min_interwake):
                short_wakes.append(i-1)
                short_wakes.append(i)
        wake_changes = pd.to_datetime([wake_changes[j] for j in set(range(len(wake_changes))) - set(short_wakes)])
        #Separate onsets and offsets of awakenings
        wake_on = [] 
        wake_off = [] 
        for i in range(len(wake_changes)): 
            if i % 2: 
                wake_off.append(wake_changes[i]) 
            else : 
                wake_on.append(wake_changes[i]) 
        #Extract awakening metrics     
        wake_df = pd.DataFrame(columns=['WASO','AwaNo'],index=index)
        df[col_wake+'_window'] = np.nan
        df[col_wake+'_window'].loc[wake_on] = 1
        wake_df['AwaNo'] = df[col_wake+'_window'].resample('24H',base=base_s).count()
        df[col_wake+'_window'].loc[wake_off] = 0
        df[col_wake+'_window'] = df[col_wake+'_window'].fillna(method='ffill')
        #Extract Wake After Sleep Onsets (minutes), adding numer of awakenings to correct for the missing min at end
        wake_df['WASO'] = df[col_wake+'_window'].resample('24H',base=base_s).sum() + wake_df['AwaNo']
        data[col_wake+'_window'] = df[col_wake+'_window']
        data[col_wake+'_window'] = data[col_wake+'_window'].fillna(method='ffill')
        return df, wake_changes, wake_df,data
    
    sleep_rec = defaultdict(dict)
    self.sleep_rec = sleep_rec
    for qtl in q_sleep:
    #Execute functions
        get_threshold_bins(night, day,'mean_hr',col_sleep , base_s, base_w, quantile=qtl ,sleep_t=True)
        get_threshold_bins(night, day,'mean_hr',col_wake, base_s, base_w, quantile=qtl,sleep_t=False)
    
        night, state_changes, sleep_df, index,df = get_sleep_windows(night, df_all,col_sleep, base_s,seq_length)
        night, wake_changes, wake_df,df = get_wake_windows(night,df_all,col_wake,col_sleep,base_s)
    
        sleep_df = pd.concat([sleep_df,wake_df], axis=1)
        #Extract sleep efficiency as (TST-WASO)/TST
        sleep_TST_delta = [(x.seconds/60) for x in sleep_df['TST'] ] 
        sleep_df['SEff'] = (sleep_TST_delta - sleep_df['WASO']) / sleep_TST_delta
        #Keep only nights with enough data (more than 10 hours available in the interval)
        sleep_df = sleep_df[night_count > 600]
            
        self.data['sleep_window_'+str(qtl)] = df_all['sleep_window']
        self.data['wake_window_'+str(qtl)] = df_all['wake_window']
        self.sleep_rec[qtl] = sleep_df
            
    return self