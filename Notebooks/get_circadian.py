# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:27:27 2020

@author: marius
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, date, time, timedelta
from collections import defaultdict
from CosinorPy import file_parser, cosinor, cosinor1
from os import path


def get_circadian(d):
    iv = defaultdict(dict)
    for idx in d.keys():
        iv_col='ENMO'
        threshold = d[idx][iv_col].quantile(0.2)
        iv[idx] = get_IV_IS(d[idx][iv_col], freq='1H',binarize=False,threshold = threshold, IV_mean=True)
        pop_iv = pd.DataFrame(columns=['IV_60','IV_m','IS_60','IS_m'],index=range(len(d.keys())))
    for idx in iv.keys():
        pop_iv.loc[idx,'IV_60'] = iv[idx][0]
        pop_iv.loc[idx,'IV_m'] = iv[idx][1]
        pop_iv.loc[idx,'IS_60'] = iv[idx][2]
        pop_iv.loc[idx,'IS_m'] = iv[idx][3]
    #Extract cosinor parameters for each subject's 'mean_hr', 'ENMO'
    pop_cos = get_cosinor(d, compare =False)
    #Extract SSA parameters
    ssa, ssa_ENMO_pv, ssa_mean_hr_pv = get_ssa_pop(d)
    ssa = get_ssa_daily(d,ssa)
    return pop_iv, pop_cos, ssa, ssa_ENMO_pv, ssa_mean_hr_pv

#Function to compute the IV and IS for each subject, for freq='1H' and average over other frequencies

def get_IV_IS(self, freq='1H', binarize=False, threshold=0,IV_mean=True):

#Gonçalves, B. S., Cavalcanti, P. R., Tavares, G. R.,
#Campos, T. F., & Araujo, J. F. (2014). Nonparametric methods in
#actigraphy: An update. Sleep science (Sao Paulo, Brazil), 7(3),158-64.
    def resampled_data(self, freq, binarize=False, threshold=0):

        if binarize is False:
            data = self
        else:
            def binarized_data(self, threshold):
             """Boolean thresholding of Pandas Series"""
            return pd.Series(np.where(self > threshold, 1, 0),index=self.index)
        
            data = binarized_data(self,threshold)

        resampled_data = data.resample(freq).sum()
        return resampled_data
    
    def intradaily_variability(data):
        r"""Calculate the intradaily variability"""

        c_1h = data.diff(1).pow(2).mean()

        d_1h = data.var()

        return (c_1h / d_1h)
    
    def interdaily_stability(data):
        r"""Calculate the interdaily stability"""

        d_24h = data.groupby([data.index.hour,data.index.minute,data.index.second]).mean().var()

        d_1h = data.var()

        return (d_24h / d_1h)
    
    data_1 = resampled_data(self,freq, binarize, threshold)
       
    IV_60 = intradaily_variability(data_1)
    IS_60 = interdaily_stability(data_1)
    
    if IV_mean==True:
        freqs=['1T', '2T', '3T', '4T', '5T', '6T', '8T', '9T', '10T',
            '12T', '15T', '16T', '18T', '20T', '24T', '30T',
            '32T', '36T', '40T', '45T', '48T', '60T']
        data = [resampled_data(self,freq, binarize, threshold) for freq in freqs]
        IV = [intradaily_variability(datum) for datum in data]
        #print(IV)
        IS = [interdaily_stability(datum) for datum in data]
        #print(IS)
        
    IV_m = np.mean(IV)
    IS_m = np.mean(IS)
    
    if IV_mean ==False:
        return IV_60, IS_60
    else:
        return IV_60, IV_m, IS_60, IS_m

#Function to get cosinor analysis for each subject, can return a summary df, or the entire dict with more parameters
def get_cosinor(d, compare=False):
    cos = defaultdict(dict)

    for idx in d.keys():
        cos[idx]['mean_hr'] = cosinor.fit_me((d[idx]['minute_of_day']+np.arange(len(d[idx])))/60, d[idx]['mean_hr'], n_components = 2, period = 24, model_type = 'lin', lin_comp = False, alpha = 0, name = '', save_to = '', plot=False, plot_residuals=False, plot_measurements=False, plot_margins=False, return_model = True, plot_phase = False)
        cos[idx]['ENMO'] = cosinor.fit_me((d[idx]['minute_of_day']+np.arange(len(d[idx])))/60, d[idx]['ENMO'], n_components = 2, period = 24, model_type = 'lin', lin_comp = False, alpha = 0, name = '', save_to = '', plot=False, plot_residuals=False, plot_measurements=False, plot_margins=False, return_model = True, plot_phase = False)
        if compare==True:
            cos[idx]['hr_vs_enmo'] = cosinor.compare_pair(X1=(d[idx]['minute_of_day']+np.arange(len(d[idx])))/60, Y1=d[idx]['mean_hr'], 
                                 X2=(d[idx]['minute_of_day']+np.arange(len(d[idx])))/60, Y2=d[idx]['ENMO'], test1 = 'mean_hr', test2 = 'ENMO', 
                                 n_components = 2, period = 24, lin_comp = True, model_type = 'lin', alpha = 0, 
                                 save_to = '', non_rhythmic = False, plot_measurements=False, plot_residuals=False)
        
    cos_pop_data = pd.DataFrame(columns=['period','amplitude_hr', 'acrophase_hr','mesor_hr','amplitude_enmo','acrophase_enmo','mesor_enmo'],index=range(len(d.keys())))

    for idx in cos.keys():
        cos_pop_data.loc[idx,'period'] = cos[idx]['mean_hr'][2]['period']
        cos_pop_data.loc[idx,'amplitude_hr'] = cos[idx]['mean_hr'][2]['amplitude']
        cos_pop_data.loc[idx,'acrophase_hr'] = cos[idx]['mean_hr'][2]['acrophase']    
        cos_pop_data.loc[idx,'mesor_hr'] = cos[idx]['mean_hr'][2]['mesor']
        cos_pop_data.loc[idx,'amplitude_enmo'] = cos[idx]['ENMO'][2]['amplitude']
        cos_pop_data.loc[idx,'acrophase_enmo'] = cos[idx]['ENMO'][2]['acrophase']    
        cos_pop_data.loc[idx,'mesor_enmo'] = cos[idx]['ENMO'][2]['mesor']
    
    return cos_pop_data

#Example return for compare_pair function: the params seem to be the coeficcients from an OLS fit model (model = sm.OLS(Y, X_fit))
#(results.pvalues[idx_params], results.params[idx_params], results)
#(array([3.21584633e-04, 3.71032098e-02, 4.03026086e-05, 1.74922581e-01]),
#array([ 2.52482527,  1.46128235,  2.87721325, -0.94775272]),
#<statsmodels.regression.linear_model.RegressionResultsWrapper at 0x211ba789448>)

#Exploring the structure of the returned wrapper from the cosinor.fit_me
#Prints the name of the model wrapper
#print(cosinor_data[0])
#dict_keys(['p', 'p_reject', 'SNR', 'RSS', 'resid_SE', 'ME'])
#print(cosinor_data[1].keys())
#dict_keys(['period', 'amplitude', 'acrophase', 'mesor'])
#print(cosinor_data[2].keys())
#print(cosinor_data[2]['acrophase'])
#looks like this is an increasing array from of 1000 values from 0 to 1000
#plt.plot(cosinor_data[3])
#plt.show()
#Looks like this is an array containing 100 values for the model fit
#plt.plot(cosinor_data[4])
#plt.show()
    
#Extracts ssa parameters for each subject - takes a long time to run, so only executes if the files don't already exist
#in the folder.

def get_SSA_par(df,L=1440): # 2 <= L <= N/2
    ### import packages ###
    import numpy as np
    from scipy import linalg # linear algebra (matrix) processing package
    import math   # math module
    ### initialize variables ###
    N=len(df)
    K=N-L+1
    ### SVD ###
    X=np.array([[df[i+j] for j in range(0,L)] for i in range(0,K)]) # trajectory matrix
    U, s, V=linalg.svd(X) # singular value decomposition (SVD) 
    l=s**2 # partial variances
    r=len(s)#np.linalg.matrix_rank(X) # matrix rank and total number of components
    ### time-series components ###
    gkList=np.zeros(shape=(r,N)) # zero matrix in whose rows SSA components will be saved
    for k in range(0,r):
            Uk=U[:,k] # k-th order column singular vector
            Vk=V[k,:] # k-th order row singular vector
            Xk=s[k]*np.outer(Uk,Vk) # k-th order matrix component
            gk=[] # empty array in which to save successive k-th order component values 
            for i in range(min(K-1,L-1),-max(K-1,L-1)-1,-1): # loop over diagonals
                gki=np.mean(np.diag(np.fliplr(Xk),i)) # successive time.series values
                gk.append(gki)
            gkList[k]=gk # k-th order component
    ### w-corr matrix ###
    w=[] # empty array to which to add successive weights
    LL=min(L,K)
    KK=max(L,K)
    for ll in range(1,LL+1): # first 1/3 part of weights
        w.append(ll)
    for ll in range(LL+1,KK+1): # second 1/3 part of weights
        w.append(LL)
    for ll in range(KK+1,N+1): # third 1/3 part of weights
        w.append(N-ll)
    kMin=kkMin=0 # show w-corr matrix for first 20 index values
    kMax=kkMax=20
    #wMatriz=np.zeros(shape=(kMin,kMax)) # initial zero matrix  
    #for k in range(kMin,kMax):
        #for kk in range(kkMin,kkMax):
            #wMatriz[k][kk]=sum(w*gkList[k]*gkList[kk])/(math.sqrt(sum(w*gkList[k]*gkList[k]))*math.sqrt(sum(w*gkList[kk]*gkList[kk])))   
    wMatrix=[[sum(w*gkList[k]*gkList[kk])/(math.sqrt(sum(w*gkList[k]*gkList[k]))*math.sqrt(sum(w*gkList[kk]*gkList[kk]))) for k in range(kMin,kMax)] for kk in range(kkMin,kkMax)]
    wMatrix=np.array(wMatrix)
    return (r, l, gkList, wMatrix); 

#Extract SSA parameters, partial variances stored in ssa_io_pv, other stored in dict ssa, read from files in folder for
#convenience, as full analysis takes a long time for each subject
def get_ssa_pop(d):
    ssa = defaultdict(dict)
    cols = ['ENMO','mean_hr']
    for col in cols:
        for idx in d.keys():
            if path.exists('ssa_'+col+'_gk_'+str(idx)+'.csv') & path.exists('ssa_'+col+'_wm_'+str(idx)+'.csv'):
                ssa[idx]['wMatrix_'+col] = pd.read_csv('ssa_'+col+'_wm_'+str(idx)+'.csv', names=range(20))
                ssa[idx]['gkList_'+col] = pd.read_csv('ssa_'+col+'_gk_'+str(idx)+'.csv', header=0)
            else:
                ssa[idx]['r'],ssa[idx]['pv'+col],ssa[idx]['gkList'+col],ssa[idx]['wMatrix'+col] = get_SSA_par(d[idx][col],L=1440)
                pd.DataFrame(ssa[idx]['gkList'+col][:10,:]).to_csv('ssa_'+col+'_gk_'+str(idx)+'.csv', index=False)
                pd.DataFrame(ssa[idx]['wMatrix'+col]).to_csv('ssa_'+col+'_wm_'+str(idx)+'.csv', index=False)
                pd.DataFrame(ssa[idx]['pv'+col]).to_csv('ssa_'+col+'_pv_'+str(idx)+'.csv', index=False)
    if path.exists('ssa_ENMO_pv.csv'):
        ssa_ENMO_pv = pd.read_csv('ssa_ENMO_pv.csv')
    if path.exists('ssa_mean_hr_pv.csv'):
        ssa_mean_hr_pv = pd.read_csv('ssa_mean_hr_pv.csv')
    return ssa, ssa_ENMO_pv, ssa_mean_hr_pv

def get_ssa_daily(d,ssa,freq='15T'):
    for idx in ssa.keys():
        enmo = np.transpose(ssa[idx]['gkList_ENMO'])
        enmo = enmo.set_index(d[idx].index)
        hr = np.transpose(ssa[idx]['gkList_mean_hr'])
        hr = hr.set_index(d[idx].index)
        #Get sum of trend and circadian components
        enmo_gk = enmo[0]+enmo[1]
        hr_gk = hr[0]+hr[1]
        #Get SSA acrophases
        ssa[idx]['acro_ENMO'] = enmo_gk.resample('24H').agg(lambda x : np.nan if x.count() == 0 else x.idxmax())
        ssa[idx]['acro_mean_hr'] = hr_gk.resample('24H').agg(lambda x : np.nan if x.count() == 0 else x.idxmax())
        #Get vectors of coarse-grain params for ML
        ssa[idx]['gksum_ENMO_'+freq] = enmo_gk.resample(freq).mean()
        ssa[idx]['gksum_mean_hr_'+freq] = hr_gk.resample(freq).mean()
        #Get trend i.e. mesor
        ssa[idx]['trend_ENMO'] = enmo[0].resample('24H').mean()
        ssa[idx]['trend_mean_hr'] = hr[0].resample('24H').mean()
        #Get period in minutes
        ssa[idx]['per_ENMO'] = (ssa[idx]['acro_ENMO'] - ssa[idx]['acro_ENMO'].shift(1)).astype('timedelta64[m]')
        ssa[idx]['per_mean_hr'] = (ssa[idx]['acro_mean_hr'] - ssa[4]['acro_mean_hr'].shift(1)).astype('timedelta64[m]')
    return ssa 
