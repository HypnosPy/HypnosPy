import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas as pd
from glob import glob
import re
import sys
import os
from scipy import stats as sps
from scipy.interpolate import interp1d

### This whole thing needs modularizing

def sleep_quality_from_wake_col(df, wake_col="wake", strategy="sleepEfficiencyAll", th_awakening_epochs=10):
    """
        This function implements different notions of sleep quality.
        For far strategy can be:
        - sleepEfficiencyAll (0-100): the percentage of time slept (wake=0) in the dataframe
        - sleepEfficiency5min (0-100): similar to above but only considers wake=1 if the awake period is longer than th_awakening (default = 10)
        - awakening (> 0): counts the number of awakenings in the period. An awakening is a sequence of wake=1 periods larger than th_awakening (default = 10)
        - awakeningIndex (> 0)
        - arousal (> 0):
        - arousalIndex (>0):
        - totalTimeInBed (in hours)
        - totalSleepTime (in hours)
        - totalWakeTime (in hours)
        - SRI (Sleep Regularity Index, in percentage %)
    """
    if strategy == "sleepEfficiencyAll":
        return 100 * (1. - df[wake_col].sum() / df.shape[0])
    
    elif strategy == "sleepEfficiency5min": # same as used in aarti's paper
        df["tmp_wake"] = df[wake_col].copy() # Avoid modifying the original values in the wake col
        df["consecutive_state"], _ = check_consecutive(df, wake_col)
        # 5 minutes = 10 entries counts
        # Change wake from 1 to 0 if group has less than 10 entries (= 5min)
        df.loc[(df["tmp_wake"] == 1) & (df["consecutive_state"] <= th_awakening_epochs), "tmp_wake"] = 0
        sleep_quality = 100 * (1. - df["tmp_wake"].sum() / df.shape[0])
        # delete aux cols
        del df["consecutive_state"]
        del df["tmp_wake"]
        return sleep_quality
    
    elif strategy == "awakening" or strategy =="awakeningIndex":
        df["consecutive_state"], df["gids"] = check_consecutive(df, wake_col)
        grps = df[(df[wake_col] == 1) & (df["consecutive_state"] > th_awakening_epochs)].groupby("gids")
        del df["consecutive_state"]
        del df["gids"]
        if strategy == "awakening":
            return len(grps)
        elif strategy == "awakeningIndex":
            totalHoursSlept = (df.shape[0]/60.)
            return len(grps) / totalHoursSlept
        else:
            ERROR____
    
    elif strategy == "arousal" or strategy == "arousalIndex":
        arousals = ((df[wake_col] == 1) & (df[wake_col] != df[wake_col].shift(1).fillna(0))).sum()
        if strategy == "arousal":
            return arousals
        elif strategy == "arousalIndex":
            totalHoursSlept = (df.shape[0]/60.)
            return arousals / totalHoursSlept
    
    elif strategy == "totalTimeInBed":
        return df.shape[0] / 60.
    
    elif strategy == "totalSleepTime":
        return ((df[wake_col] == 0).sum()) / 60.
    
    elif strategy == "totalWakeTime":
        return (df[wake_col].sum()) / 60.
    
    elif strategy == "SRI":
        # SRI needs some tuning
        #df = data.between_time(bed_time,wake_time, include_start = True, 
                                            #include_end = True) Change this to sleep search window
        sri_delta = np.zeros(len(df[df.index[0]:df.shift(periods=-1,freq='D').index[-1]]))
        for i in range(len(df[df.index[0]:df.shift(periods=-1,freq='D').index[-1]])):
            if df[sleep_col][df.index[i]] == df[sleep_col].shift(periods=-1,freq='D')[df.index[i]]:
                sri_delta[i] = 1
            else:
                sri_delta[i] = 0
        sri_df = pd.DataFrame(sri_delta)
        sri = -100 + (200 / (len(df[df.index[0]:df.shift(periods=-1,freq='D').index[-1]]))) * sri_df.sum()
        return float(sri)
    
    else:
        ERROR____
        print("Strategy %s is unknown." % (strategy))


# %%
def get_sleep_metrics(df_in,
                      min_sleep_block,
                      sleep_block_col,
                      sleep_metrics,
                      wake_sleep_col="Wake_Sleep",
                      timecol="ts", 
                      what_is_sleep_value=False # The default is sleep = 0, set this to True if in your dataset sleep = 1
                      ):
    
    df = df_in[df_in[sleep_block_col] >= min_sleep_block].copy()
    
    if what_is_sleep_value:
        df[wake_sleep_col] = ~df[wake_sleep_col]
    
    series = []
    for sleep_metric in sleep_metrics:
        r = df[[sleep_block_col,
                wake_sleep_col]].groupby(sleep_block_col).apply(lambda grp: sleep_quality_from_wake_col(grp,
                                                                                                wake_sleep_col,
                                                                                                strategy=sleep_metric))
        r.name = sleep_metric
        series.append(r)
    
    return pd.concat(series, axis=1).reset_index()
