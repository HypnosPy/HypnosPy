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

def label_sleep(self, q_sleep=[0.4], bed_time = '20:00',wake_time = '12:00', lengths = [15], vol_threshold=5, take_SRI=True):
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
    
    SRI = defaultdict(dict)
    self.SRI=SRI
    
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
        df[col_sleep+'_vard'] = df['mean_hr'].rolling(10,center=True).std()
        data[col_sleep+'_vard'] = data['mean_hr'].rolling(10,center=True).std()
        df[col_sleep+'_labels'] = (df[col_sleep+'_bin']==1.0) & (df[col_sleep+'_len'] > min_window_length) #& df[col_sleep+'_vard']<0.20
        df[col_sleep+'_labels'] = (df[col_sleep+'_labels'] == True).astype(int)
        #Extract the times when labelled sleep changes to wake and viceversa
        state_changes = df[col_sleep+'_labels'].diff().fillna(0)[lambda x: x != 0].index.tolist()
        state_changes = pd.to_datetime(state_changes).to_frame()
        
        df[col_sleep+'_windowb'] = np.nan
        for i in range(len(state_changes)): 
            if i % 2: 
                df[col_sleep+'_windowb'].loc[state_changes.iloc[i]] = 0
            else : 
                df[col_sleep+'_windowb'].loc[state_changes.iloc[i]] = 1
        df[col_sleep+'_windowb'][0] = 0        
        df[col_sleep+'_windowb'] = df[col_sleep+'_windowb'].fillna(method='ffill')
        data[col_sleep+'_windowb'] = df[col_sleep+'_windowb']
        data[col_sleep+'_windowb'][0] = 0
        data[col_sleep+'_windowb'] = data[col_sleep+'_windowb'].fillna(method='ffill').fillna(method='bfill')
        sleep_over = []
        state_filtered = state_changes.copy()
        data[col_sleep+'_naps'] = np.nan  
        #state_date = pd.to_datetime(state_changes)
        for i in np.arange(1, len(state_changes)-1,2):
            if (data[state_changes[0][i]-timedelta(hours=4):state_changes[0][i]][col_sleep+'_windowb'].sum()<180) or \
                (data[state_changes[0][i]-timedelta(hours=1):state_changes[0][i]][col_sleep+'_windowb'].sum()<40):
                sleep_over.append(i-1)
                sleep_over.append(i)
                data[col_sleep+'_naps'].loc[state_changes[0][i]] = 0
                data[col_sleep+'_naps'].loc[state_changes[0][i-1]] = 1
        data[col_sleep+'_naps'][0] = 0        
        data[col_sleep+'_naps'] = data[col_sleep+'_naps'].fillna(method='ffill')
        
        keep = set(range(len(state_filtered))) - set(sleep_over)
        state_filtered = state_filtered.iloc[list(keep)]
        #print(state_filtered)
        

        #Extract index of nights available for the subject to pass onto sleep df
        index = state_changes.resample('24H',base=base_s).min().index
        sleep_df = pd.DataFrame(columns=['TST','sleep_onset','sleep_offset','weekday'], index = index)
        #Extract sleep onset each night as earliest wake-sleep transition after base hour
        sleep_df['sleep_onset'] = state_changes.resample('24H',base=base_s).min()
        
        
        #Extract sleep onset each night as last sleep-wake transition after base hour
        sleep_df['sleep_offset'] = state_filtered.resample('24H',base=base_s).max()
        #sleep_df['sleep_offset'] = state_changes.resample('24H',base=base_s).max()
        
        #print(sleep_df)
        for i in range(len(sleep_df['sleep_onset'])):
            lookon = data[sleep_df['sleep_onset'][i]-timedelta(hours=4):sleep_df['sleep_onset'][i]+timedelta(minutes=60)]
            new_onset_list = lookon[lookon[col_sleep+'_vard']>vol_threshold].index.tolist()
            if len(new_onset_list)!=0:
                sleep_df['sleep_onset'][i] = new_onset_list[-1]
        for j in range(len(sleep_df['sleep_offset'])):
            lookoff = data[sleep_df['sleep_offset'][j]-timedelta(minutes=1):sleep_df['sleep_offset'][j]+timedelta(hours=2)]
            new_offset_list = lookoff[lookoff[col_sleep+'_vard']>vol_threshold].index.tolist()
            if len(new_offset_list)!=0:
                sleep_df['sleep_offset'][j] = new_offset_list[0]
                
        sleep_df['TST']= sleep_df['sleep_offset'] -  sleep_df['sleep_onset']
        
        sleep_df['weekday'] = sleep_df.index.dayofweek
        sleep_df = pd.DataFrame(sleep_df).dropna()
        #Label sleep windows on original df according to sleep_df
        
        data[col_sleep+'_window'] = np.nan
        data[col_sleep+'_window'].loc[sleep_df['sleep_onset']] = 1
        data[col_sleep+'_window'].loc[sleep_df['sleep_offset']] = 0
        data[col_sleep+'_window'][0] = 0
        data[col_sleep+'_window'] = data[col_sleep+'_window'].fillna(method='ffill').fillna(method='bfill')
        
        #data[col_sleep+'_napbuffer'] = np.nan
        #data[col_sleep+'_napbuffer'].loc[sleep_df['sleep_onset']] = 1
        #data[col_sleep+'_napbuffer'].loc[sleep_df['sleep_offset']] = 2
        #data[col_sleep+'_napbuffer'][0] = 2
        #data[col_sleep+'_napbuffer'] = data[col_sleep+'_napbuffer'].fillna(method='ffill').fillna(method='bfill')
        
        data[col_sleep+'_naps'] = ((data[col_sleep+'_naps']==1)&(data[col_sleep+'_window']==0)& \
                                   (data[col_sleep+'_window'].rolling(180).sum()==0)& \
                                   (data[col_sleep+'_window'].shift(-90).rolling(90).sum()==0)).astype(int)
        
        sleep_df['NapDuration'] = data[col_sleep+'_naps'].resample('24H',base=base_s).sum()
            
        return df, state_changes, sleep_df, index, data

    def get_wake_windows(df,data,col_wake, col_sleep,base_s):
        #min_len = 5
        max_len = 60
        min_interwake = 10
        df[col_wake+'_window'] = (df[col_wake+'_bin']==0.0) & (df[col_wake+'_len'] < max_len) & (df[col_wake+'_len'] >= 5) & (data[col_sleep+'_window']==1)    
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
        data[col_wake+'_window'][0] = 0
        data[col_wake+'_window'] = data[col_wake+'_window'].fillna(method='ffill')#.fillna(method='bfill')
        return df, wake_changes, wake_df,data
    
    #Function to get the sleep regularity index
    def get_SRI(self, quantile=0.425, lens=40):
        sleep_col = 'sleep_window_'+str(qtl)+'_'+str(lens)
        sri_delta = np.zeros(len(self.data[self.data.index[0]:self.data.shift(periods=-1,freq='D').index[-1]]))
        for i in range(len(self.data[self.data.index[0]:self.data.shift(periods=-1,freq='D').index[-1]])):
            if self.data[sleep_col][self.data.index[i]] == self.data[sleep_col].shift(periods=-1,freq='D')[self.data.index[i]]:
                sri_delta[i] = 1
            else:
                sri_delta[i] = 0
        sri_df = pd.DataFrame(sri_delta)
        sri = -100 + (200 / (len(self.data[self.data.index[0]:self.data.shift(periods=-1,freq='D').index[-1]]))) * sri_df.sum()
        self.SRI[qtl][lens] = float(sri)
        return self 
    
    sleep_rec = defaultdict(dict)
    self.sleep_rec = sleep_rec
    for qtl in q_sleep:
        for lens in lengths:
    #Execute functions
            get_threshold_bins(night, day,'mean_hr',col_sleep , base_s, base_w, quantile=qtl ,sleep_t=True)
            get_threshold_bins(night, day,'mean_hr',col_wake, base_s, base_w, quantile=qtl,sleep_t=False)
    
            night, state_changes, sleep_df, index,df = get_sleep_windows(night, df_all,col_sleep, base_s,lens)
            night, wake_changes, wake_df,df = get_wake_windows(night,df_all,col_wake,col_sleep,base_s)
    
            sleep_df = pd.concat([sleep_df,wake_df], axis=1)
            #Extract sleep efficiency as (TST-WASO)/TST
            sleep_TST_delta = [(x.seconds/60) for x in sleep_df['TST'] ] 
            sleep_df['SEff'] = (sleep_TST_delta - sleep_df['WASO']) / sleep_TST_delta
            #Keep only nights with enough data (more than 10 hours available in the interval)
            sleep_df = sleep_df[night_count > 600]
            
            self.data['sleep_window_'+str(qtl)+'_'+str(lens)] = df_all['sleep_window']
            self.data['wake_window_'+str(qtl)+'_'+str(lens)] = df_all['wake_window']
            self.data['sleep_windowb_'+str(qtl)+'_'+str(lens)] = df_all['sleep_windowb']
            self.data['naps_'+str(qtl)+'_'+str(lens)] = df[col_sleep+'_naps']
            self.sleep_rec[qtl][lens] = sleep_df
            if take_SRI:
                get_SRI(self,quantile=qtl, lens=lens)
            
    return self

#Limbs are 'dw','ndw' and 'thigh'
def get_vanhees(self, limb = 'dw',q_sleep=0.1, bed_time = '20:00',wake_time = '12:00', min_len = 30, gaps=60, factor=15):
    if all(self.data.ts.diff()[1:] == np.timedelta64(60, 's')) == False:
        vhees= self.data.resample('1T').mean().copy()
    else:
        vhees = self.data.copy()
    day_start = '08:00'
    #day_end = '23:00'
    col_sleep = 'sleep'
    col_wake = 'wake'
    #Base is read from function input and start of day
    base_s = int(bed_time[0:2])
    base_w = int(day_start[0:2])
    sleep_rec = defaultdict(dict)
    if hasattr(self,'sleep_recvh')==0:
        self.sleep_recvh=sleep_rec
    
    def get_threshold_bins(vhees, col_apply, col_new, base_w,quantile):
        vhees[col_apply+'_diff'] = vhees[col_apply].diff().abs()
        vhees[col_apply+'_5mm'] = vhees[col_apply+'_diff'].rolling(5).median()
        vhees[col_apply+'_10pct'] = vhees[col_apply+'_5mm'].resample('24H',base=base_w).apply(lambda x: np.quantile(x,q_sleep))
        vhees[col_apply+'_10pct'] = vhees[col_apply+'_10pct'].ffill(axis=0)
        vhees[col_apply+'_10pct'] = vhees[col_apply+'_10pct'].fillna(method='bfill')
        vhees[col_apply+'_bin'] =np.where((vhees[col_apply+'_5mm']-vhees[col_apply+'_10pct']*factor)>0, 0, 1)
        
        vhees[col_apply+'_seq_id'] = vhees.groupby((vhees[col_apply+'_bin'] != 
                                                            vhees[col_apply+'_bin'].shift(1).fillna(False)).cumsum()).ngroup()
        vhees[col_apply+'_len'] = 0
        vhees[col_apply+'_len'] = vhees[[col_apply+'_len', col_apply+'_seq_id']].groupby(col_apply+'_seq_id').transform('count')
        return vhees
    
    def get_sleep_windows(df,data,limb,night_count,cols, base_w, base_s,min_len):
        for col_apply in cols:
            df[col_apply+'_labels'] = (df[col_apply+'_bin']==1.0) & (df[col_apply+'_len'] > min_len).astype(int)
            df[col_apply+'_gaps'] = (df[col_apply+'_bin']==0.0) & (df[col_apply+'_len'] > gaps).astype(int)
            df[col_apply+'_labels_2'] = ((df[col_apply+'_labels']==1.0) & (df[col_apply+'_gaps']==0.0)).astype(int)
        df['sleepvote_'+limb] = ((df[cols[0]+'_labels_2']==1.0) & (df[cols[1]+'_labels_2']==1.0)).astype(int)
        #Extract the times when labelled sleep changes to wake and viceversa
        state_changes = df['sleepvote_'+limb].diff().fillna(0)[lambda x: x != 0].index.tolist()
        state_changes = pd.to_datetime(state_changes).to_frame()
        #print(state_changes)
        #Extract index of nights available for the subject to pass onto sleep df
        #Keep only nights with enough data (more than 7 hours available in the interval) 
        index = state_changes.resample('24H',base=base_s).min()[night_count > 420].index
        #print(state_changes.resample('24H',base=base_s).min()[night_count > 480])
        sleep_df = pd.DataFrame(columns=['TST','sleep_onset','sleep_offset','weekday'], index = index)
        #print(sleep_df)
        #Extract sleep onset each night as earliest wake-sleep transition after base hour
        sleep_df['sleep_onset'] = state_changes.resample('24H',base=base_s).min()[night_count > 420]
        #Extract sleep onset each night as last sleep-wake transition after base hour
        sleep_df['sleep_offset'] = state_changes.resample('24H',base=base_s).max()
        sleep_df['TST']= sleep_df['sleep_offset'] -  sleep_df['sleep_onset']
        sleep_df['weekday'] = sleep_df.index.dayofweek
        sleep_df = pd.DataFrame(sleep_df)
        #Label sleep windows on original df according to sleep_df
        df['sleep_window_'+limb] = np.nan
        sleep_df = sleep_df.dropna()


        #print(sleep_df)
        df['sleep_window_'+limb].loc[sleep_df['sleep_onset']] = 1
        df['sleep_window_'+limb].loc[sleep_df['sleep_offset']] = 0
        df['sleep_window_'+limb] = df['sleep_window_'+limb].fillna(method='ffill').fillna(method='bfill')
        data['sleep_window_'+limb] = df['sleep_window_'+limb]
        data['sleep_window_'+limb] = data['sleep_window_'+limb].fillna(method='ffill')
        data['sleep_window_'+limb] = data['sleep_window_'+limb].fillna(method='bfill')
        return df, state_changes, sleep_df, index,data

    #Execute functions
    get_threshold_bins(vhees,'pitch_mean_'+limb,col_sleep, base_w, quantile=q_sleep)
    get_threshold_bins(vhees,'roll_mean_'+limb,col_sleep, base_w, quantile=q_sleep)

    night = vhees.between_time(bed_time,wake_time, include_start = True, 
                                            include_end = True)
    night_count = night['pitch_mean_'+limb].resample('24H',base=base_s).count()
    
    night, state_changes, sleep_df, index,vhees= get_sleep_windows(night,vhees,limb,night_count,['pitch_mean_'+limb,'roll_mean_'+limb],
                                                                       base_w,base_s,min_len)
    #night, state_changes_r, sleep_df_r, index,vhees = get_sleep_windows(night,vhees,limb,night_count,'roll_mean_'+limb, base_w,base_s,min_len)
       
    #fig, ax = plt.subplots(2,1,figsize=(20,15))
    #ax[0].plot(vhees.pitch_mean_dw)
    #ax[0].plot(vhees['pitch_mean_dw'+'_labels_2']*10, color='black', label='labels')
    #ax[0].plot(vhees['pitch_mean_dw'+'_window']*20, color='green', label='sleep')
    #ax[0].plot(vhees['pitch_mean_dw'+'_gaps']*-10, color='red', label='gaps')
    
    #ax[1].plot(vhees.roll_mean_dw)
    #ax[1].plot(vhees['roll_mean_dw'+'_labels_2']*10, color='black', label='labels')
    #ax[1].plot(vhees['roll_mean_dw'+'_window']*20, color='green', label='sleep')
    #ax[1].plot(vhees['roll_mean_dw'+'_gaps']*-10, color='red', label='gaps')
    #plt.legend()
    #plt.show()
    
    #Extract sleep efficiency as (TST-WASO)/TST
    sleep_TST_delta = [(x.seconds/60) for x in sleep_df['TST'] ] 
            
    self.data['sleep_window_'+limb] = vhees['sleep_window_'+limb]
    
    self.sleep_recvh[limb] = sleep_df
            
    return self

    #Function to get the sleep regularity index
    #def get_SRI(self, q_sleep = 0.425, lengths = 40):
    #    for qtl in q_sleep:
    #        for lens in lengths:
    #            sleep_col = 'sleep_window_'+str(qtl)+str(lens)
    #            sri_delta = np.zeros(len(self.data[self.data.index[0]:self.data.shift(periods=-1,freq='D').index[-1]]))
    #            for i in range(len(self.data[self.data.index[0]:self.data.shift(periods=-1,freq='D').index[-1]])):
    #                if self.data[sleep_col][self.data.index[i]] == self.data[sleep_col].shift(periods=-1,freq='D')[self.data.index[i]]:
    #                    sri_delta[i] = 1
    #                else:
    #                    sri_delta[i] = 0
    #                    sri_df = pd.DataFrame(sri_delta)
    #                    sri = -100 + (200 / (len(self.data[self.data.index[0]:self.data.shift(periods=-1,freq='D').index[-1]]))) * sri_df.sum()
    #                    self.SRI[qtl][lens] = float(sri)
    #    return self 