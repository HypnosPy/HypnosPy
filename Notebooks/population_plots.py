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
from datetime import timedelta
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
    import matplotlib.dates as dates
    DFList = [pd.DataFrame(group[1]) for group in d[i].data.groupby(d[i].data.index.day)]
    fig, ax1 = plt.subplots(len(DFList), 1, figsize= (14, 10))
    #ax1.set_title("Physical activity and sedentary time per hour")                        
    for idx in range(len(DFList)):
        #Resampling: hourly
        df2_h = DFList[idx]
        ax1[idx].tick_params(axis='x', which='both',bottom=False,top=False, labelbottom=True, rotation=0)
        ax1[idx].set_facecolor('snow')
        #ax1[idx].set_xlabel('Time')
        ax1[idx].set_ylim(0,max(d[i].data['ENMO']/10))
        ax1[idx].set_xlim(DFList[idx].index[0] - timedelta(hours=DFList[idx].index[0].hour) - 
                          timedelta(minutes=DFList[idx].index[0].minute) - timedelta(seconds=DFList[idx].index[0].second), 
                          DFList[idx].index[0] - timedelta(hours=DFList[idx].index[0].hour) - 
                          timedelta(minutes=DFList[idx].index[0].minute) - timedelta(seconds=DFList[idx].index[0].second) +
                          timedelta(minutes=1439))

        ax1[idx].plot(df2_h.index, df2_h['ENMO']/10,label='ENMO', linewidth=1, color ='black',alpha=1)
        ax1[idx].fill_between(df2_h.index, 0, df2_h['min_VigPA']*100,facecolor ='darkgreen',alpha=0.7, label='VPA',edgecolor='darkgreen')
        ax1[idx].fill_between(df2_h.index, 0, df2_h['min_MVPA']*100,facecolor ='forestgreen',alpha=0.7, label='MVPA',edgecolor='forestgreen')
        
        ax1[idx].fill_between(df2_h.index, 0, df2_h['min_LPA']*100,facecolor ='lightgreen',alpha=0.7, label='LPA',edgecolor='lightgreen')
        ax1[idx].fill_between(df2_h.index, 0, (df2_h['MET_Sed'])*100,facecolor ='palegoldenrod',alpha=0.6,label='sedentary',edgecolor='palegoldenrod')
        ax1[idx].fill_between(df2_h.index, 0, -(df2_h['sleep_window_0.4']-2)*100,facecolor ='royalblue',alpha=1,label='sleep')
        ax1[idx].fill_between(df2_h.index, 0, (df2_h['wake_window_0.4'])*150,facecolor ='cyan',alpha=1,label='wake')
        #ax1[idx].fill_between(df2_h.index, 0, (1-df2_h['crespo'])*10,facecolor ='cyan',alpha=0.1,label='crespo_rest')
        
        #ax1[idx].xaxis.set_minor_locator(dates.HourLocator(interval=4))   # every 4 hours
        #ax1[idx].xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes
        #ax1[idx].xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
        #ax1[idx].xaxis.set_major_formatter(dates.DateFormatter('\n%d-%m-%Y'))
        ax1[idx].set_xticks([])
        ax1[idx].set_yticks([])
    #    for ax in ax_row:
        # create a twin of the axis that shares the x-axis
        ax2 = ax1[idx].twinx()
        #ax2.set_ylabel('mean_HR')  # we already handled the x-label with ax1
        ax2.plot(df2_h.index, df2_h['mean_hr'],label='HR', color='red')
        ax2.set_ylim(30,max(d[i].data['mean_hr']))
        ax2.set_xticks([])
        ax2.set_yticks([])
              
    ax1[-1].xaxis.set_minor_locator(dates.HourLocator(interval=4))   # every 4 hours
    ax1[-1].xaxis.set_minor_formatter(dates.DateFormatter('%H:%M')) # hours and minutes

    handles, labels = ax1[-1].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles+handles2, labels+labels2, loc='lower center',ncol=4)
    return fig

def plot_circ(d,nonlinear_na):

    fig, ax1 = plt.subplots(2, 2, figsize= (20, 12))


    ax1[0,0].plot(d[0].data.index,np.transpose(d[0].ssa['ENMO']['gk'])[0]+np.transpose(d[0].ssa['ENMO']['gk'])[1], color='red',label='ENMO SSA')
    ax1[0,0].plot(d[0].data.index, d[0].data['ENMO']/5,color='blue',alpha=0.4,label='ENMO')
    ax1[0,0].set_ylim(0,max(d[0].data['ENMO']/5))
    ax1[0,0].set_ylabel('ENMO (scaled)')
    ax1[0,0].set_xlim('2005-07-11','2005-07-14')
    ax1[0,0].set_xticks([])
    ax1[0,0].set_yticks([])

    for i in range(len(d[0].ssa['ENMO']['acrophase'])):
        ax1[0,0].axvline(x=d[0].ssa['ENMO']['acrophase'][i],color='orange')

    ax1[1,0].plot(d[0].data.index, d[0].data['mean_hr'],color='green',alpha=0.4,label='HR')
    ax1[1,0].plot(d[0].data.index,np.transpose(d[0].ssa['mean_hr']['gk'])[0]+np.transpose(d[0].ssa['mean_hr']['gk'])[1],color='red',label='HR SSA')
    ax1[1,0].set_ylim(40,max(d[0].data['mean_hr']))
    ax1[1,0].set_ylabel('HR')
    ax1[1,0].set_yticks(ax1[1,0].get_yticks()[::2])
    ax1[1,0].xaxis.set_minor_locator(dates.HourLocator(interval=8)) 
    ax1[1,0].xaxis.set_minor_formatter(dates.DateFormatter('%H'))
    ax1[1,0].xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
    ax1[1,0].xaxis.set_major_formatter(dates.DateFormatter('\n%A'))
    ax1[1,0].set_xlim('2005-07-11','2005-07-14')

    for i in range(len(d[0].ssa['mean_hr']['acrophase'])):
        ax1[1,0].axvline(x=d[0].ssa['mean_hr']['acrophase'][i],color='orange')
    
    for idx in range(2):
        for i in range(len(d[0].sleep_rec)):
            ax1[idx,0].axvspan(d[0].sleep_rec['sleep_onset'][i],d[0].sleep_rec['sleep_offset'][i],facecolor='grey',alpha=0.2)

    dfa_cols = ['dfa_ENMO_sleep','dfa_ENMO_wake','dfa_HR_sleep','dfa_HR_wake','dfa_HRV_sleep','dfa_HRV_wake']
    sampen_cols = ['se_ENMO_sleep','se_ENMO_wake','se_HR_sleep','se_HR_wake','se_HRV_sleep','se_HRV_wake']
    dfa_labels = ['ENMO_s','ENMO_w','HR_s','HR_w','HRV_s','HRV_w']
    sampen_labels = ['ENMO_s','ENMO_w','HR_s','HR_w','HRV_s','HRV_w']
    dfa = nonlinear_na[dfa_cols]
    dfa_t = np.transpose(dfa)
    sampen = nonlinear_na[sampen_cols]
    sampen_t = np.transpose(sampen)
    ax1[0,1].boxplot(dfa_t, labels= dfa_labels)
    ax1[0,1].tick_params(axis='x', which='both',bottom=False,top=False, labelbottom=True, rotation=0)
    ax1[0,1].set_yticks(ax1[0,1].get_yticks()[::2])

    ax1[1,1].boxplot(sampen_t, labels = sampen_labels)
    ax1[1,1].tick_params(axis='x', which='both',bottom=False,top=False, labelbottom=True, rotation=0)
    ax1[1,1].set_yticks(ax1[1,1].get_yticks()[::2])
    ax1[1,1].set_ylim(0,2)

    ax1[0,0].title.set_text('Circadian analysis - ENMO')
    ax1[1,0].title.set_text('Circadian analysis - HR')
    ax1[0,1].title.set_text('DFA')
    ax1[1,1].title.set_text('Sample entropy')

    fig.legend(bbox_to_anchor=(0.4,0.8), ncol=2)
    #plt.tight_layout()
    return fig

def plot_circ2(d):

    fig, ax1 = plt.subplots(2, 2, figsize= (20, 14))
    sns.set_palette(sns.light_palette("green"))
    cmap = sns.light_palette("green")
    
    ax1[0,0].plot(d[0].data.index,np.transpose(d[0].ssa['ENMO']['gk'])[0]+np.transpose(d[0].ssa['ENMO']['gk'])[1], color='red',label='ENMO SSA')
    ax1[0,0].plot(d[0].data.index, d[0].data['ENMO']/5,color='blue',alpha=0.4,label='ENMO')
    ax1[0,0].set_ylim(0,max(d[0].data['ENMO']/5))
    ax1[0,0].set_ylabel('ENMO (scaled)')
    ax1[0,0].set_xlim('2005-07-11','2005-07-14')
    ax1[0,0].set_xticks([])
    ax1[0,0].set_yticks([])

    for i in range(len(d[0].ssa['ENMO']['acrophase'])):
        ax1[0,0].axvline(x=d[0].ssa['ENMO']['acrophase'][i],color='orange')

    ax1[1,0].plot(d[0].data.index, d[0].data['mean_hr'],color='green',alpha=0.4,label='HR')
    ax1[1,0].plot(d[0].data.index,np.transpose(d[0].ssa['mean_hr']['gk'])[0]+np.transpose(d[0].ssa['mean_hr']['gk'])[1],color='red',label='HR SSA')
    ax1[1,0].set_ylim(40,max(d[0].data['mean_hr']))
    ax1[1,0].set_ylabel('HR')
    ax1[1,0].set_yticks(ax1[1,0].get_yticks()[::2])
    ax1[1,0].xaxis.set_minor_locator(dates.HourLocator(interval=8)) 
    ax1[1,0].xaxis.set_minor_formatter(dates.DateFormatter('%H'))
    ax1[1,0].xaxis.set_major_locator(dates.DayLocator(interval=1))    # every day
    ax1[1,0].xaxis.set_major_formatter(dates.DateFormatter('\n%A'))
    ax1[1,0].set_xlim('2005-07-11','2005-07-14')

    for i in range(len(d[0].ssa['mean_hr']['acrophase'])):
        ax1[1,0].axvline(x=d[0].ssa['mean_hr']['acrophase'][i],color='orange')
    
    for idx in range(2):
        for i in range(len(d[0].sleep_rec)):
            ax1[idx,0].axvspan(d[0].sleep_rec['sleep_onset'][i],d[0].sleep_rec['sleep_offset'][i],facecolor='grey',alpha=0.2)
    
    #ax1[0,1].scatter(d_pop_t['ARI'],d_pop_t['SRI'])
    s1 = sns.scatterplot(d_pop_t['ARI'],d_pop_t['SRI'],hue = d_pop_t['VO2max'],
                    cmap=cmap,ax=ax1[0,1])
    #s1.legend_.remove()
    ax1[0,1].tick_params(axis='x', which='both',bottom=False,top=False, labelbottom=True, rotation=0)
    ax1[0,1].set_yticks(ax1[0,1].get_yticks()[::2])
    ax1[0,1].set_xticks(ax1[0,1].get_xticks()[::2])
    ax1[0,1].set_xlabel('ARI')
    ax1[0,1].set_ylabel('SRI')

    #ax1[1,1].scatter(d_pop_t['ENMO_phisleep_delay'].astype('timedelta64[s]')//60,d_pop_t['ENMO_SSA_per'])
    s2 = sns.scatterplot(d_pop_t['ENMO_phisleep_delay'].astype('timedelta64[s]')//60,d_pop_t['ENMO_SSA_per'],
                    ax=ax1[1,1], cmap=cmap, hue=d_pop_t['VO2max'])
    s2.legend_.remove()
    ax1[1,1].tick_params(axis='x', which='both',bottom=False,top=False, labelbottom=True, rotation=0)
    ax1[1,1].set_yticks(ax1[1,1].get_yticks()[::2])
    ax1[1,1].set_xticks(ax1[1,1].get_xticks()[::2])
    ax1[1,1].set_xlabel('Peak PA - sleep onset delay (min)')
    ax1[1,1].set_ylabel('Peak PA period (min)')

    ax1[0,0].title.set_text('Circadian analysis - ENMO')
    ax1[1,0].title.set_text('Circadian analysis - HR')
    ax1[0,1].title.set_text('Activity - Sleep regularity distribution')
    ax1[1,1].title.set_text('Sleep and activity periods')

    handles, labels = ax1[0,0].get_legend_handles_labels()
    handles2, labels2 = ax1[1,0].get_legend_handles_labels()
    fig.legend(handles+handles2, labels+labels2, bbox_to_anchor=(0.4,0.8),ncol=2)
    #plt.tight_layout()
    return fig