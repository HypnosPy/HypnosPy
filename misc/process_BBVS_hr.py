import os
import tempfile
from glob import glob
import pandas as pd
import numpy as np

from hypnospy import Wearable
from hypnospy.data import RawProcessing
from hypnospy.analysis import TimeSeriesProcessing
from hypnospy import Experiment
from hypnospy import Diary
from hypnospy.analysis import SleepMetrics
from multiprocessing import Pool

from sklearn.metrics import mean_squared_error, cohen_kappa_score

def load_experiment(data_path, diary_path, start_hour):

    # Configure an Experiment
    exp = Experiment()

    # Iterates over a set of files in a directory.
    # Unfortunately, we have to do it manually with RawProcessing because we are modifying the annotations
    for file in glob(data_path):
        pp = RawProcessing(file,
                           # HR information
                           col_for_hr="mean_hr",
                           # Activity information
                           cols_for_activity=["stdMET_highIC_Branch"],
                           is_act_count=False,
                           device_location="dw",
                           # Datetime information
                           col_for_datetime="REALTIME",
                           strftime="%d-%b-%Y %H:%M:%S",
                           # Participant information
                           col_for_pid="id")
        pp.data["hyp_act_x"] = pp.data["hyp_act_x"] - 1.0  # adjust for the BBVA dataset

        w = Wearable(pp)  # Creates a wearable from a pp object
        exp.add_wearable(w)

    # Set frequency for every wearable in the collection
    exp.set_freq_in_secs(60)

    # Changing the hour the experiment starts from midnight (0) to 3pm (15)
    exp.change_start_hour_for_experiment_day(start_hour)

    diary = Diary().from_file(diary_path)
    exp.add_diary(diary)

    return exp

def one_loop(hparam):

    exp_id, data_path, diary_path, start_hour, end_hour, hr_quantile, hr_merge_blocks, hr_min_window_length = hparam

    print("Q:", hr_quantile, "L:", hr_min_window_length, "G", hr_merge_blocks)

    exp = load_experiment(data_path, diary_path, start_hour)

    tsp = TimeSeriesProcessing(exp)
    tsp.fill_no_activity(0.0001)
    tsp.detect_non_wear(strategy="none")
    tsp.check_valid_days(min_activity_threshold=-100000, max_non_wear_minutes_per_day=180, check_sleep_period=False,
                         check_diary=True)
    tsp.drop_invalid_days()

    tsp.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hr", hr_quantile=hr_quantile,
                                hr_volarity_threshold=hr_volarity, hr_rolling_win_in_minutes=5,
                                hr_sleep_search_window=(start_hour, end_hour),
                                hr_min_window_length_in_minutes=hr_min_window_length,
                                hr_volatility_window_in_minutes=10, hr_merge_blocks_gap_time_in_min=hr_merge_blocks,
                                hr_sleep_only_in_sleep_search_window=True, hr_only_largest_sleep_period=True)

    # tsp.detect_sleep_boundaries(strategy="adapted_van_hees", output_col="hyp_sleep_period_vanheesndw",
    #                             vanhees_cols=["pitch_mean_ndw", "roll_mean_ndw"], vanhees_start_hour=vanhees_start_hour,
    #                             vanhees_quantile=vanhees_quantile, vanhees_minimum_len_in_minutes=vanhees_window_length,
    #                             vanhees_merge_tolerance_in_minutes=vanhees_time_merge_block)

    sm = SleepMetrics(exp)
    sm_results = sm.get_sleep_quality(strategy="sleepEfficiency", sleep_period_col="hyp_sleep_period_hr",
                                      wake_sleep_col="hyp_sleep_period_hr")
    sm_results = sm.get_sleep_quality(strategy="sri", sleep_period_col=None,
                                      wake_sleep_col="hyp_sleep_period_hr")

    print(sm_results)

    df_acc = []
    mses = {}
    cohens = {}

    #print("Calculating evaluation measures...")
    for w in exp.get_all_wearables():

        diary_sleep = w.data[w.diary_sleep].astype(int)
        hr_sleep = w.data["hyp_sleep_period_hr"].astype(int)

        if diary_sleep.shape[0] == 0 or hr_sleep.shape[0] == 0:
            continue

        mses["diary_hr"] = mean_squared_error(diary_sleep, hr_sleep)
        cohens["diary_hr"] = cohen_kappa_score(diary_sleep, hr_sleep)

        tst_diary = w.get_total_sleep_time_per_day(based_on_diary=True)
        tst_hr = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")

        onset_diary = w.get_onset_sleep_time_per_day(based_on_diary=True)
        onset_diary.name = "onset_diary"
        onset_hr = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")
        onset_hr.name = "onset_hr"

        offset_diary = w.get_offset_sleep_time_per_day(based_on_diary=True)
        offset_diary.name = "offset_diary"
        offset_hr = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")
        offset_hr.name = "offset_hr"

        df_res = pd.concat((onset_hr, onset_diary,
                            offset_hr, offset_diary,
                            tst_diary, tst_hr), axis=1)

        df_res["pid"] = w.get_pid()
        for comb in ["diary_hr"]:
            df_res["mse_" + comb] = mses[comb]
            df_res["cohens_" + comb] = cohens[comb]

        # View signals
        # w.change_start_hour_for_experiment_day(0)
        w.view_signals(["activity", "hr", "sleep", "diary"], sleep_cols=["hyp_sleep_period_hr"])

        df_acc.append(df_res)

    exp_id += 1
    df_acc = pd.concat(df_acc)
    df_acc["exp_id"] = exp_id
    df_acc["quantile"] = hr_quantile
    df_acc["window_lengths"] = hr_min_window_length
    df_acc["time_merge_blocks"] = hr_merge_blocks

    df_acc.to_csv(os.path.join(output_path, "exp_%d.csv" % (exp_id)), index=False)


if __name__ == "__main__":

    diary_path = "./data/diaries/BBVS_new_diary.csv" # <- TODO: CHANGE HERE
    data_path = "./data/small_collection_bvs/BVS*.csv" # <- TODO: CHANGE HERE

    hr_volarity = 5
    exp_id = 0
    quantiles = np.arange(0.1, 0.96, 0.025)
    window_lengths = np.arange(10, 121, 5)
    time_merge_blocks = [-1]  # [30, 60, 120, 240, 360, 420]

    start_hour = 15
    end_hour = 15


    hparam_list = []
    exp_id = 0
    for hr_quantile in quantiles:
        for hr_merge_blocks in time_merge_blocks:
            for hr_min_window_length in window_lengths:

                exp_id += 1
                hparam_list.append([exp_id, data_path, diary_path, start_hour, end_hour,
                                    hr_quantile, hr_merge_blocks, hr_min_window_length])

    with tempfile.TemporaryDirectory() as output_path:
        #pool = Pool(processes=8)
        #pool.map(one_loop, hparam_list)
        one_loop(hparam_list[0])

        # Cleaning up: Merge all experiments and
        dfs = glob(output_path + "/*")
        bigdf = pd.concat([pd.read_csv(f) for f in dfs])
        bigdf.to_csv("results_BBVS_fullday_noG_Aug28.csv.gz", index=False)

    print("DONE!")

