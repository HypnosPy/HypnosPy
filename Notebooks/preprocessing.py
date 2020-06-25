# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:51:54 2020

@author: marius
"""
#Preprocessing
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
from collections import defaultdict

class Subject:
    
    def __init__(self, filename):
        self.filename = filename
        self.data = pd.read_csv(r"./Data/"+filename)
        #Set index
        if 'real_time' in self.data.columns:
            self.data["ts"] = [datetime.strptime(ts,"%d-%m-%Y %H:%M:%S") for ts in self.data["real_time"]] 
        elif 'REALTIME' in self.data.columns:    
            self.data["ts"] = [datetime.strptime(ts,"%d-%b-%Y %H:%M:%S") for ts in self.data["REALTIME"]] 
        self.data['date'] = self.data['ts'].dt.date
        self.data['hour'] = self.data['ts'].dt.hour
        self.data.set_index(pd.DatetimeIndex(self.data['ts']), inplace=True)

        #Set ENMO
        self.data['ENMO'] = (self.data["ACC"]/0.0060321) + 0.057
        #Get other variables
        if 'predRMR_Oxford2005' in self.data.columns:
            self.RMR = self.data['predRMR_Oxford2005'][0]
        if 'P_TR_FITNESS_HighPt_est' in self.data.columns:
            self.VO2max = self.data['P_TR_FITNESS_HighPt_est'][0]
        self.age = self.data['age'][0]
        self.weight = self.data['weight'][0]
        self.height = self.data['height'][0]
        self.BMI = self.weight / (self.height **2)
        
            
    def __str__(self):
        return "Subject: {}".format(self.filename)
    
    def get_METS(self, sed = 1.5, MVPA = 3, vig = 6):
        if 'stdMET_highIC_Branch' in self.data.columns:
            #Sedentary activities counted from {sed} METs
            self.data['MET_Sed'] = self.data['stdMET_highIC_Branch'].apply(lambda x : 1 if x <= sed-1 else 0)
            #WHO recommendations would count moderate PA as between 3-6 METs
            self.data['MET_MVPA'] = self.data['stdMET_highIC_Branch'].apply(lambda x : x if x > MVPA-1 else 0)
            self.data['min_MVPA'] = self.data['stdMET_highIC_Branch'].apply(lambda x : 1 if x > MVPA-1 else 0)
            #WHO recommendations would count moderate PA as >6 METs
            self.data['MET_VigPA'] = self.data['stdMET_highIC_Branch'].apply(lambda x : x if x > vig-1 else 0)
            self.data['min_VigPA'] = self.data['stdMET_highIC_Branch'].apply(lambda x : 1 if x > vig-1 else 0)
            #LPA
            self.data['min_LPA'] = self.data['stdMET_highIC_Branch'].apply(lambda x : 1 if ((x >= sed-1) and (x < MVPA-1)) else 0)
        else:
            print("No METs column in data!")
        return self
        
    def get_HRV(self):
        #Get HRV
        file = self.data
        file = file[file.PWEAR>0] #remove buffer no_wear
        if 'real_time' in file.columns:
            file['real_time'] = pd.to_datetime(file['real_time'], dayfirst=True) #datetime pandas format
        elif 'REALTIME' in file.columns:
            file['REALTIME'] = pd.to_datetime(file['REALTIME'], dayfirst=True) #datetime pandas format
        file = file[file.PWEAR>0] #remove buffer heart rate
        #calculate HRV from IBI (data is corrupted so we should flag/not use that datapoint)
        file['hrv_ms'] = np.where(file.min_ibi_2_in_milliseconds != 1992.0,file['max_ibi_2_in_milliseconds']-file['min_ibi_1_in_milliseconds'], np.nan)
        #file.hrv_milliseconds.fillna(file.hrv_milliseconds.mean(), inplace=True) #fill HRV nans with mean
        file.fillna(file.mean(), inplace=True) #fill nans with mean and if completely empty fill with 0
        file.fillna(0, inplace=True)
        self.data['hrv_ms'] = file['hrv_ms']
        
        return self
    
    def get_HRV_profile(self):
        hrv = defaultdict(dict)
        file = self.data
        file['sleep_window_0.4'] = file['sleep_window_0.4'].fillna(method='bfill')
        #Calculate profile for wake times
        file_wake = file.loc[file['sleep_window_0.4']==1]
        hrv['wake']['mean'] = file_wake['hrv_ms'].mean()
        hrv['wake']['std'] = file_wake['hrv_ms'].std()
        hrv['wake']['skew'] = file_wake['hrv_ms'].skew()
        hrv['wake']['kurtosis'] = file_wake['hrv_ms'].kurtosis()
        #Calculate profile for sleep windows
        file_sleep = file.loc[(file['sleep_window_0.4']==2) & (file['wake_window_0.4']==0)]
        hrv['sleep']['mean'] = file_sleep['hrv_ms'].mean()
        hrv['sleep']['std'] = file_sleep['hrv_ms'].std()
        hrv['sleep']['skew'] = file_sleep['hrv_ms'].skew()
        hrv['sleep']['kurtosis'] = file_sleep['hrv_ms'].kurtosis()
        #Calculate profile for MVPA
        file_MVPA = file[file['min_MVPA']==1]
        hrv['MVPA']['mean'] = file_MVPA['hrv_ms'].mean()
        hrv['MVPA']['std'] = file_MVPA['hrv_ms'].std()
        hrv['MVPA']['skew'] = file_MVPA['hrv_ms'].skew()
        hrv['MVPA']['kurtosis'] = file_MVPA['hrv_ms'].kurtosis()
        #Calculate profile for sedentary time
        file_sed = file[file['MET_Sed']==1]
        hrv['sed']['mean'] = file_sed['hrv_ms'].mean()
        hrv['sed']['std'] = file_sed['hrv_ms'].std()
        hrv['sed']['skew'] = file_sed['hrv_ms'].skew()
        hrv['sed']['kurtosis'] = file_sed['hrv_ms'].kurtosis()
        
        self.HRV = hrv
        return self
        
        
    
    def get_PA(self, wake_time='07:00', bed_time='23:00'):
        #Extracting daily physical activity data (not adjusted to full days yet)
        #Set time interval
        day = self.data.between_time(wake_time,bed_time, include_start = True, 
                                            include_end = True)
        #Resampling: daily
        day_rd = day.resample('D', base=7).sum()
        #Gets PA averages
        self.MVPA_mean = day_rd['MET_MVPA'].mean()
        self.MVPA_std = day_rd['MET_MVPA'].std()
        self.VigPA_mean = day_rd['MET_VigPA'].mean()
        self.VigPA_std = day_rd['MET_VigPA'].std()
        self.MVPAmins_mean = day_rd['min_MVPA'].mean()
        self.MVPAmins_std = day_rd['min_MVPA'].std()
        self.VigPAmins_mean = day_rd['min_VigPA'].mean()
        self.VigPAmins_std = day_rd['min_VigPA'].std()
        self.LPAmins_mean = day_rd['min_LPA'].mean()
        self.LPAmins_std = day_rd['min_LPA'].std()
        self.Sed_mean = day_rd['MET_Sed'].mean()
        self.Sed_std = day_rd['MET_Sed'].std()
        self.VigPA_dcount = np.count_nonzero(day_rd['MET_VigPA'])
        self.pa_rec = day_rd[['ENMO','MET_Sed','MET_MVPA','MET_VigPA','min_MVPA','min_VigPA','min_LPA']]
        return self
    
    from sleep_analysis import get_sleep, get_SRI
    from circadian_analysis import get_IV_IS, get_cosinor, get_SSA,get_SSA_par
    from nonlinear_analysis import get_nonlinear, get_nonlin_params
    from crespo_analysis import Crespo

    def get_windows(self):
        df_copy = self.data.copy()
        df_copy['sleep_cumsum'] = (df_copy['sleep_window_0.4']-1).cumsum()

        df_night = pd.DataFrame(df_copy.loc[lambda df_copy: df_copy['sleep_cumsum'].diff() == 0])
        df_day = pd.DataFrame(df_copy.loc[lambda df_copy: df_copy['sleep_cumsum'].diff() == 1])

        self.sleep_windows = [pd.DataFrame(group[1]) for group in df_night.groupby(df_night['sleep_cumsum'])]
        self.wake_windows = [pd.DataFrame(group[1]) for group in df_day.groupby(df_day['timepoint']//1440)]
        
        for idx in range(len(self.sleep_windows)):
            self.sleep_windows[idx]['length'] = len(self.sleep_windows[idx])
        for idx in range(len(self.wake_windows)):
            self.wake_windows[idx]['length'] = len(self.wake_windows[idx])
        return self
    
    def get_daily_stats(self):
        sleep_rec = self.sleep_rec.copy()
        pa_rec = self.pa_rec.copy()
        ssa = self.ssa.copy()
    
        nonlin_cols = ['dfa_ENMO_sleep','dfa_ENMO_wake','dfa_HR_sleep','dfa_HR_wake','dfa_HRV_sleep','dfa_HRV_wake',
                       'se_ENMO_sleep','se_ENMO_wake','se_HR_sleep','se_HR_wake','se_HRV_sleep','se_HRV_wake']
        nonlinear = pd.DataFrame(columns=nonlin_cols,index=range(len(self.nonlinear['hrv_ms']['wake'].keys())))
        #nonlinear = pd.DataFrame(columns=nonlin_cols,index=self.pa_rec.index)

        for idx in self.nonlinear['ENMO']['sleep'].keys():
            nonlinear.loc[idx,'dfa_ENMO_sleep'] = self.nonlinear['ENMO']['sleep'][idx]['DFA']
            nonlinear.loc[idx,'dfa_HR_sleep'] = self.nonlinear['mean_hr']['sleep'][idx]['DFA']
            nonlinear.loc[idx,'dfa_HRV_sleep'] = self.nonlinear['hrv_ms']['sleep'][idx]['DFA']
            nonlinear.loc[idx,'se_ENMO_sleep'] = self.nonlinear['ENMO']['sleep'][idx]['SampEn']
            nonlinear.loc[idx,'se_HR_sleep'] = self.nonlinear['mean_hr']['sleep'][idx]['SampEn']
            nonlinear.loc[idx,'se_HRV_sleep'] = self.nonlinear['hrv_ms']['sleep'][idx]['SampEn']
   
        for jdx in self.nonlinear['hrv_ms']['wake'].keys():
            nonlinear.loc[jdx,'dfa_ENMO_wake'] = self.nonlinear['ENMO']['wake'][jdx]['DFA']
            nonlinear.loc[jdx,'dfa_HR_wake'] = self.nonlinear['mean_hr']['wake'][jdx]['DFA']
            nonlinear.loc[jdx,'dfa_HRV_wake'] = self.nonlinear['hrv_ms']['wake'][jdx]['DFA']
            nonlinear.loc[jdx,'se_ENMO_wake'] = self.nonlinear['ENMO']['wake'][jdx]['SampEn']
            nonlinear.loc[jdx,'se_HR_wake'] = self.nonlinear['mean_hr']['wake'][jdx]['SampEn']
            nonlinear.loc[jdx,'se_HRV_wake'] = self.nonlinear['hrv_ms']['wake'][jdx]['SampEn']
            
       
        nonlinear = nonlinear.set_index(pa_rec.index)
        #print(nonlinear)
        wake_lengths = pd.DataFrame(columns = ['w_length'],index = pa_rec.index)
        for idx in range(len(self.wake_windows)):
            wake_lengths.iloc[idx]['w_length'] = self.wake_windows[idx]['length'][0]
        sleep_lengths = pd.DataFrame(columns = ['s_length'],index = self.pa_rec.index)
        for idx in range(len(self.sleep_windows)):
            sleep_lengths.iloc[idx]['s_length'] = self.sleep_windows[idx]['length'][0]

        sleep_rec.index = pd.to_datetime(self.sleep_rec.index.values) - timedelta(hours=20)
        #print(df.sleep_rec)
        daily = pd.concat([pa_rec,sleep_rec, wake_lengths,sleep_lengths,nonlinear],axis=1)

        daily['ENMO_SSA_phi'] = self.ssa['ENMO']['acrophase']
        daily['mean_hr_SSA_phi'] = self.ssa['mean_hr']['acrophase']
        daily['ENMO_SSA_per'] = self.ssa['ENMO']['period']
        daily['mean_hr_SSA_per'] = self.ssa['mean_hr']['period']
        daily['ENMO_phisleep_delay'] = daily['sleep_onset'] - daily['ENMO_SSA_phi']
        daily['mean_hr_phisleep_delay'] = daily['sleep_onset'] - daily['mean_hr_SSA_phi']

        self.daily_stats = daily
        return self