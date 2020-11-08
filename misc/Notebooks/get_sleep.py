# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:30:10 2020

@author: marius
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, date, time, timedelta
from os import path
from collections import defaultdict

#Function to compute the sleep windows for each subject, computed using the simple HR heuristic
#takes arguments for the start and finish times of the sleep analysis (at the moment does not look at day naps)
#takes the length of the shortest sequence deemed as sleep and the quantile for the sleep threshold

def extract_sleep(d, quantiles, bed_time = '20:00',wake_time = '12:00', seq_length = 15,take_sri=False):
    sleep = defaultdict(dict)
    sleep_pop = pd.DataFrame(index=range(len(d.keys())))
    for idx in d.keys():
        for q in quantiles:
            sleep[idx][q] = pd.DataFrame()
            #Applies function to get PA data only from wake_time to bed_time
            sleep[idx][q] = get_tst(d[idx], bed_time = bed_time,wake_time = wake_time,seq_length = seq_length, q_sleep=q,take_sri=False)
            sleep_pop.loc[idx,'TST_avg'+str(q)] = sleep[idx][q]['TST'].mean().total_seconds()/60
            sleep_pop.loc[idx,'WASO'+str(q)] = sleep[idx][q]['WASO'].mean()
            sleep_pop.loc[idx,'AwaNo'+str(q)] = sleep[idx][q]['AwaNo'].mean()
            sleep_pop.loc[idx,'SEff'+str(q)] = sleep[idx][q]['SEff'].mean()
            if take_sri==True:
                sleep_pop.loc[idx,'SRI'+q] = sri
    return sleep, sleep_pop

def get_tst(df, bed_time = '20:00',wake_time = '12:00',seq_length = 20, q_sleep=0.4,take_sri=False, col_sleep='sleep', col_wake='wake'):
    if all(df.ts.diff()[1:] == np.timedelta64(60, 's')) == False:
        df = df.resample('1T').mean()
    
    night = df.between_time(bed_time,wake_time, include_start = True, 
                                            include_end = True)
    day_start = '09:00'
    day_end = '22:00'
    day = df.between_time(day_start,day_end, include_start = True, 
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
    
    def get_sleep_windows(df, col_sleep, base_s, min_window_length):
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
        
        return df, state_changes, sleep_df, index
    
    def get_wake_windows(df, col_wake, base_s,min_len = 5, max_len = 60, min_interwake = 10):
        df[col_wake+'_window'] = (df[col_wake+'_bin']==0.0) & (df[col_wake+'_len'] < max_len) & (df[col_wake+'_len'] >= min_len) & (df[col_sleep+'_window']==1)    
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
        night[col_wake+'_window'].loc[wake_on] = 1
        wake_df['AwaNo'] = night[col_wake+'_window'].resample('24H',base=base_s).count()
        df[col_wake+'_window'].loc[wake_off] = 0
        df[col_wake+'_window'] = df[col_wake+'_window'].fillna(method='ffill')
        #Extract Wake After Sleep Onsets (minutes), adding numer of awakenings to correct for the missing min at end
        wake_df['WASO'] = df[col_wake+'_window'].resample('24H',base=base_s).sum() + wake_df['AwaNo']
        return df, wake_changes, wake_df
    
    #Execute functions
    get_threshold_bins(night, day,'mean_hr', col_sleep, base_s, base_w, quantile=q_sleep ,sleep_t=True)
    get_threshold_bins(night, day, 'mean_hr', col_wake,base_s, base_w, quantile=q_sleep,sleep_t=False)
    
    night, state_changes, sleep_df, index = get_sleep_windows(night, col_sleep, base_s,seq_length)
    night, wake_changes, wake_df = get_wake_windows(night, col_wake,base_s,min_len = 5, max_len = 60)
    
    sleep_df = pd.concat([sleep_df,wake_df], axis=1)
    #Extract sleep efficiency as (TST-WASO)/TST
    sleep_TST_delta = [(x.seconds/60) for x in sleep_df['TST'] ] 
    sleep_df['SEff'] = (sleep_TST_delta - sleep_df['WASO']) / sleep_TST_delta
    #Keep only nights with enough data (more than 10 hours available in the interval)
    sleep_df = sleep_df[night_count > 600]
    
    #Visualisation aid
    #plt.plot(night.index,night['hr_threshold_labels']*200, color='blue')
    #plt.plot(night.index,night['mean_hr']-night['hr_threshold'], color='orange')
    #plt.plot(night.index,night['hr_threshold_window']*250,color='black')
    #plt.scatter(night.index,night['wake_threshold_window']*400, color='magenta')
    #for i in range(len(sleep_df)):
    #        plt.axvspan(sleep_df['sleep_onset'][i],sleep_df['sleep_offset'][i],facecolor='grey',alpha=0.4)
    #plt.show()
    
    if take_sri == True:
        sri = get_sri(night,col_sleep+'_window')
        return sleep_df,sri
    else: 
        return sleep_df

#Function to get the sleep regularity index
def get_sri(df, sleep_col):
            sri_delta = np.zeros(len(df[df.index[0]:df.shift(periods=-1,freq='D').index[-1]]))
            for i in range(len(df[df.index[0]:df.shift(periods=-1,freq='D').index[-1]])):
                if df[sleep_col][df.index[i]] == df[sleep_col].shift(periods=-1,freq='D')[df.index[i]]:
                    sri_delta[i] = 1
                else:
                    sri_delta[i] = 0
            sri_df = pd.DataFrame(sri_delta)
            sri = -100 + (200 / (len(df[df.index[0]:df.shift(periods=-1,freq='D').index[-1]]))) * sri_df.sum()
            return float(sri)     

