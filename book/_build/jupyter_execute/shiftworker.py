# Shiftworker analysis

from hypnospy import Wearable
from hypnospy.data import RawProcessing
from hypnospy.analysis import Viewer, NonWearingDetector, SleepBoudaryDetector, SleepWakeAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pp = RawProcessing("./data/shiftworker/Shiftworker.csv",
                           # HR information
                           col_for_hr="mean_hr",
                           # Activity information
                           cols_for_activity=["ACC"],
                           is_act_count=True,
                           # Datetime information
                           col_for_datetime="real_time",
                           strftime="%d/%m/%Y %H:%M",
                           # Participant information
                           col_for_pid="id")

w = Wearable(pp)

#Check out data columns
#print(w.data.head(10))

#Define parameters fo HR-based sleep algorithm
hr_quantile = 0.4
hr_min_window_length = 60
hr_merge_blocks = 180
hr_volarity = 5

#Time to consider as start and end of each experiment day - if equal the sleep labelling occurs
#over the entire 24 hours
start_hour = 18
end_hour = 18

# Label sleep using HypnosPy HR algorithms

sbd = SleepBoudaryDetector(w)

sbd.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hr", hr_quantile=hr_quantile,
                                hr_volarity_threshold=hr_volarity, hr_rolling_win_in_minutes=5,
                                hr_sleep_search_window=(start_hour, end_hour),
                                hr_min_window_length_in_minutes=hr_min_window_length,
                                hr_volatility_window_in_minutes=10, hr_merge_blocks_gap_time_in_min=hr_merge_blocks,
                                hr_sleep_only_in_sleep_search_window=True, hr_only_largest_sleep_period=True)

#Plot sleep labels together with HR and acitivty signals
v = Viewer(w)

v.view_signals(["activity", "hr", "sleep"],
                sleep_cols=["hyp_sleep_period_hr"],
              alphas={'sleep': 0.3})