# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:27:33 2020

@author: mariu
"""

import pandas as pd
from collections import defaultdict
from preprocessing import Subject
import seaborn as sns
import numpy as np
import timedelta
from scipy import stats as sps
import matplotlib.pyplot as plt
import matplotlib.dates as dates

def plot_basic(d,frequency='H'):
    fig, ax1 = plt.subplots(len(d.keys()), 1, figsize= (14, 15))
    #ax1.set_title("Physical activity and sedentary time per hour")                        
    for idx in d.keys():
        #Resampling: hourly
        df2_h = d[idx].data.resample(frequency).sum()
        ax1[idx].tick_params(axis='x', which='both',bottom=False,top=False, labelbottom=True, rotation=10)
        ax1[idx].set_xlabel('Time, sleep windows shaded grey')
        ax1[idx].set_ylim(0,max(df2_h['MET_MVPA']))

        ax1[idx].grid(color='#b2b2b7', linestyle='--',linewidth=1, alpha=0.5)
        ax1[idx].plot(df2_h.index,df2_h['MET_MVPA'], label='MET_MVPA',linewidth=3, color ='green',alpha=1)
        ax1[idx].plot(df2_h.index,df2_h['MET_VigPA'], label='MET_VigPA',linewidth=3, color ='red',alpha=1)
        ax1[idx].set_ylabel('MET_MVPA')
        ax1[idx].legend()
        
        #Add grey windows for sleep
        for i in range(len(d[idx].sleep_rec)):
            ax1[idx].axvspan(d[idx].sleep_rec['sleep_onset'][i],d[idx].sleep_rec['sleep_offset'][i],facecolor='grey',alpha=0.4)
        for j in range(len(d[idx].crespo_on)):
            ax1[idx].axvspan(d[idx].crespo_on[j],d[idx].crespo_off[j],facecolor='blue', alpha=0.3)

def plot_daily(d,i):
    DFList = [pd.DataFrame(group[1]) for group in d[i].data.groupby(d[i].data.index.day)]
    fig, ax1 = plt.subplots(len(DFList), 1, figsize= (14, 25))
    #ax1.set_title("Physical activity and sedentary time per hour")                        
    for idx in range(len(DFList)):
        #Resampling: hourly
        df2_h = DFList[idx]
        ax1[idx].tick_params(axis='x', which='both',bottom=False,top=False, labelbottom=True, rotation=0)
        ax1[idx].set_facecolor('lightsteelblue')
        #ax1[idx].set_xlabel('Time')
        ax1[idx].set_ylim(0,max(d[i].data['MET_MVPA']))
        ax1[idx].set_xlim(DFList[idx].index[0] - timedelta(hours=DFList[idx].index[0].hour) - 
                          timedelta(minutes=DFList[idx].index[0].minute) - timedelta(seconds=DFList[idx].index[0].second), 
                          DFList[idx].index[0] - timedelta(hours=DFList[idx].index[0].hour) - 
                          timedelta(minutes=DFList[idx].index[0].minute) - timedelta(seconds=DFList[idx].index[0].second) +
                          timedelta(minutes=1439))

        #ax1[idx].grid(color='#b2b2b7', linestyle='--',linewidth=1, alpha=0.5)
        ax1[idx].plot(df2_h.index, df2_h['MET_MVPA'],label='MVPA_METs', linewidth=1, color ='black',alpha=1)
        ax1[idx].fill_between(df2_h.index, 0, df2_h['min_MVPA']*10,facecolor ='green',alpha=0.7, label='MVPA',edgecolor='green')
        #ax1[idx].plot(df2_h.index,df2_h['MET_VigPA'], label='MET_VigPA',linewidth=1, color ='black',alpha=1)
        ax1[idx].fill_between(df2_h.index, 0, df2_h['min_VigPA']*10,facecolor ='red',alpha=0.7, label='VPA',edgecolor='red')
        
        ax1[idx].fill_between(df2_h.index, 0, (df2_h['MET_Sed'])*10,facecolor ='sandybrown',alpha=0.6,label='sedentary',edgecolor='sandybrown')
        ax1[idx].fill_between(df2_h.index, 0, -(df2_h['sleep_window_0.4']-2)*10,facecolor ='grey',alpha=1,label='sleep')
        #ax1[idx].fill_between(df2_h.index, 0, (1-df2_h['crespo'])*10,facecolor ='cyan',alpha=0.1,label='crespo_rest')
        ax1[idx].set_ylabel('MET_MVPA')
        
        ax1[idx].xaxis.set_minor_locator(dates.HourLocator(interval=4))   # every 4 hours
        ax1[idx].xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes
        ax1[idx].xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
        ax1[idx].xaxis.set_major_formatter(dates.DateFormatter('\n%d-%m-%Y'))
        
        #ax1[idx].fill_between(df2_h.index, 0, df2_h['MET_MVPA'])
    
    handles, labels = ax1[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.3, 0.9),ncol=2)
    plt.tight_layout()
    return fig