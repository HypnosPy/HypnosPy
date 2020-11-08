# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:18:51 2020

@author: mariu
"""

import pandas as pd
from collections import defaultdict
from preprocessing import Subject
import seaborn as sns
import numpy as np
from scipy import stats as sps
import matplotlib.pyplot as plt

def get_ranks(d, shift=0, plot_ranks=True):
    ranks = defaultdict(dict)
    ranks_agg = pd.DataFrame()
    for idx in d.keys():
        if shift ==0:
            rank_m = pd.concat([d[idx].sleep_rec,d[idx].pa_rec],axis=1).resample('D').max()
        elif shift ==-1:
            rank_m = pd.concat([d[idx].sleep_rec,d[idx].pa_rec.shift(shift)],axis=1).resample('D').max()
        #print(rank_m)
        rank_m2 = rank_m.rank(method='max',ascending=False)
        ranks[idx] = rank_m2.corr(method='spearman')
        ranks_agg = pd.concat([ranks_agg, rank_m],axis=0)
    #print(ranks_agg)
    ranks_agg = ranks_agg.dropna()
    ranks_agg2 = ranks_agg.rank(method='max',ascending=False)
    ranks_agg3 = ranks_agg2.corr(method='spearman')
    
    ranks_agg_p = pd.DataFrame()  # Matrix of p-values
    for x in ranks_agg2.columns:
        for y in ranks_agg2.columns:
            corr = sps.pearsonr(ranks_agg2[x], ranks_agg2[y])
            ranks_agg_p.loc[x,y] = corr[1]
    
    if plot_ranks==True:
        plt.rcParams['figure.figsize'] = (14,9)
        sns.heatmap(ranks_agg3, annot=True)
        plt.show()
        sns.heatmap(np.round(ranks_agg_p,4), annot=True)
        plt.show()
    
    return ranks, ranks_agg3, ranks_agg_p

