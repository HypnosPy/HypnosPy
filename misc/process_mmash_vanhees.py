import tempfile
import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from hypnospy import Wearable, Diary
from hypnospy.data import RawProcessing
from hypnospy.analysis import TimeSeriesProcessing
from hypnospy import Experiment

from sklearn.metrics import mean_squared_error, cohen_kappa_score
from multiprocessing import Pool


def load_experiment(data_path, diary_path, start_hour):
    # Configure an Experiment
    exp = Experiment()

    # Iterates over a set of files in a directory.
    # Unfortunately, we have to do it manually with RawProcessing because we are modifying the annotations
    for file in glob(data_path):
        pp = RawProcessing(file,
                           # HR information
                           col_for_hr="HR",
                           # Activity information
                           cols_for_activity=["Axis1", "Axis2", "Axis3"],
                           is_act_count=False,
                           # Datetime information
                           col_for_datetime="time",
                           strftime="%Y-%b-%d %H:%M:%S",
                           # Participant information
                           col_for_pid="pid")

        w = Wearable(pp)  # Creates a wearable from a pp object
        exp.add_wearable(w)

    # Set frequency for every wearable in the collection
    # exp.set_freq_in_secs(5)

    # Changing the hour the experiment starts from midnight (0) to 3pm (15)
    exp.change_start_hour_for_experiment_day(start_hour)

    diary = Diary().from_file(diary_path)
    exp.add_diary(diary)

    return exp


def one_loop(hparam):

    exp_id, data_path, diary_path, start_hour, end_hour, vh_quantile, vh_merge_blocks, vh_min_window_length = hparam

    print("Q:", vh_quantile, "W:", vh_min_window_length, "T", vh_merge_blocks)

    exp = load_experiment(data_path, diary_path, start_hour)

    tsp = TimeSeriesProcessing(exp)
    tsp.fill_no_activity(-0.0001)
    tsp.detect_non_wear(strategy="none")
    tsp.check_valid_days(min_activity_threshold=0, max_non_wear_minutes_per_day=180, check_sleep_period=False,
                         check_diary=True)
    tsp.drop_invalid_days()

    tsp.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hr", hr_quantile=0.35,
                                hr_volarity_threshold=hr_volarity, hr_rolling_win_in_minutes=5,
                                hr_sleep_search_window=(start_hour, end_hour), hr_min_window_length_in_minutes=32.50,
                                hr_volatility_window_in_minutes=10, hr_merge_blocks_gap_time_in_min=240,
                                hr_sleep_only_in_sleep_search_window=True, hr_only_largest_sleep_period=True)

    tsp.detect_sleep_boundaries(strategy="adapted_van_hees", output_col="hyp_sleep_period_vanhees", angle_cols=[],
                                angle_use_triaxial_activity=True, angle_start_hour=start_hour,
                                angle_quantile=vh_quantile, angle_minimum_len_in_minutes=vh_min_window_length,
                                angle_merge_tolerance_in_minutes=vh_merge_blocks)

    tsp.detect_sleep_boundaries(strategy="adapted_van_hees", output_col="hyp_sleep_period_vanheespr",
                                angle_cols=["pitch", "roll"], angle_use_triaxial_activity=False,
                                angle_start_hour=start_hour, angle_quantile=vh_quantile,
                                angle_minimum_len_in_minutes=vh_min_window_length,
                                angle_merge_tolerance_in_minutes=vh_merge_blocks)

    df_acc = []
    mses = {}
    cohens = {}

    print("Calculating evaluation measures...")
    for w in exp.get_all_wearables():

        if w.data.empty:
            print("Data for PID %s is empty!" % w.get_pid())
            continue

        sleep = {}
        sleep["diary"] = w.data[w.diary_sleep].astype(int)
        sleep["hr"] = w.data["hyp_sleep_period_hr"].astype(int)
        sleep["vanhees"] = w.data["hyp_sleep_period_vanhees"].astype(int)
        sleep["vanheespr"] = w.data["hyp_sleep_period_vanheespr"].astype(int)

        if sleep["diary"].shape[0] == 0:
            continue

        for comb in ["diary_hr", "diary_vanhees", "hr_vanhees", "diary_vanheespr", "hr_vanheespr", "vanhees_vanheespr"]:
            a, b = comb.split("_")
            mses[comb] = mean_squared_error(sleep[a], sleep[b])
            cohens[comb] = cohen_kappa_score(sleep[a], sleep[b])

        tst_diary = w.get_total_sleep_time_per_day(based_on_diary=True)
        tst_hr = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")
        tst_vanhees = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_vanhees")
        tst_vanheespr = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_vanheespr")

        onset_diary = w.get_onset_sleep_time_per_day(based_on_diary=True)
        onset_diary.name = "onset_diary"
        onset_hr = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")
        onset_hr.name = "onset_hr"
        onset_vanhees = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanhees")
        onset_vanhees.name = "onset_vanhees"
        onset_vanheespr = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanheespr")
        onset_vanheespr.name = "onset_vanheespr"

        offset_diary = w.get_offset_sleep_time_per_day(based_on_diary=True)
        offset_diary.name = "offset_diary"
        offset_hr = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")
        offset_hr.name = "offset_hr"
        offset_vanhees = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanhees")
        offset_vanhees.name = "offset_vanhees"
        offset_vanheespr = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanheespr")
        offset_vanheespr.name = "offset_vanheespr"

        df_res = pd.concat((onset_hr, onset_diary, onset_vanhees, onset_vanheespr,
                            offset_hr, offset_diary, offset_vanhees, offset_vanheespr,
                            tst_diary, tst_hr, tst_vanhees, tst_vanheespr), axis=1)

        df_res["pid"] = w.get_pid()
        for comb in ["diary_hr", "diary_vanhees", "hr_vanhees", "diary_vanheespr", "hr_vanheespr", "vanhees_vanheespr"]:
            df_res["mse_" + comb] = mses[comb]
            df_res["cohens_" + comb] = cohens[comb]

        # View signals
        # w.change_start_hour_for_experiment_day(0)
        # w.view_signals(["activity", "hr", "sleep", "diary"],
        #                sleep_cols=["hyp_sleep_period_hr", "hyp_sleep_period_vanhees", "hyp_sleep_period_vanheespr"])

        df_acc.append(df_res)

    exp_id += 1
    df_acc = pd.concat(df_acc)
    df_acc["exp_id"] = exp_id
    df_acc["quantile"] = vh_quantile
    df_acc["window_lengths"] = vh_min_window_length
    df_acc["time_merge_blocks"] = vh_merge_blocks

    df_acc.to_csv(os.path.join(output_path, "exp_%d.csv" % (exp_id)), index=False)

if __name__ == "__main__":

    diary_path = "./data/diaries/mmash_diary.csv"
    data_path = "./data/collection_mmash/*.csv"

    hr_volarity = 5
    exp_id = 0
    # Best values Q: 0.35, W: 32.50, T: 240.00
    quantiles = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275,
                 0.30, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675,
                 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]
    time_merge_blocks = [30, 60, 120, 240, 360, 420]
    window_lengths = [25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50]
    start_hour = 15
    end_hour = start_hour - 1

    hparam_list = []
    exp_id = 0
    for vh_quantile in quantiles:
        for vh_merge_blocks in time_merge_blocks:
            for vh_min_window_length in window_lengths:

                exp_id += 1
                hparam_list.append([exp_id, data_path, diary_path, start_hour, end_hour,
                                    vh_quantile, vh_merge_blocks, vh_min_window_length])

    with tempfile.TemporaryDirectory() as output_path:
        pool = Pool(processes=9)
        pool.map(one_loop, hparam_list)
        #one_loop(hparam_list[0])

        # Cleaning up: Merge all experiments and
        dfs = glob(output_path + "/*")
        bigdf = pd.concat([pd.read_csv(f) for f in dfs])
        bigdf.to_csv("results_MMASH_vanhees.csv.gz", index=False)

    print("DONE!")

