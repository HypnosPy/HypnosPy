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
from scipy import stats as sps
import matplotlib.pyplot as plt

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
