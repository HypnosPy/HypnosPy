import os
import tempfile
from glob import glob
import pandas as pd
import numpy as np

from hypnospy import Wearable, Diary
from hypnospy.data import RawProcessing
from hypnospy.analysis import NonWearingDetector, SleepBoudaryDetector, Validator, Viewer
from hypnospy import Experiment
from sklearn.metrics import mean_squared_error, cohen_kappa_score

import ray
ray.init(log_to_driver=False)


def load_experiment(data_path, diary_path, start_hour):

    # Configure an Experiment
    exp = Experiment()

    # Iterates over a set of files in a directory.
    # Unfortunately, we have to do it manually with RawProcessing because we are modifying the annotations
    print("Running %s" % data_path)
    for file in glob(data_path):
        print("FILE", file)
        pp = RawProcessing(file, cols_for_activity=["stdMET_highIC_Branch"], is_act_count=False,
                           col_for_datetime="REALTIME", strftime="%d-%b-%Y %H:%M:%S", col_for_pid="id",
                           col_for_hr="mean_hr", device_location="dw")
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

@ray.remote
def one_loop(hparam):

    exp_id, data_path, diary_path, start_hour, end_hour, hr_quantile, hr_merge_blocks, hr_min_window_length, \
    hr_volarity, start_night, end_night, output_path = hparam

    print("Q:", hr_quantile, "L:", hr_min_window_length, "G", hr_merge_blocks)

    exp = load_experiment(data_path, diary_path, start_hour)
    exp.fill_no_activity(-10000)  # Maybe this needs to be changed or removed

    va = Validator(exp)
    va.remove_wearables_without_diary()

    va.flag_epoch_null_cols(["pitch_mean_ndw", "roll_mean_ndw", "pitch_mean_dw", "roll_mean_dw", "pitch_mean_thigh", "roll_mean_thigh"])

    va.flag_epoch_physical_activity_less_than(min_activity_threshold=-1000)  # Maybe this needs to be changed or removed
    va.flag_epoch_null_cols(col_list=["hyp_act_x"])
    va.flag_day_max_nonwearing(max_non_wear_minutes_per_day=3 * 60)

    va.flag_day_if_invalid_epochs_larger_than(max_invalid_minutes_per_day=5 * 60)
    va.flag_day_without_diary()

    n_removed_days = va.remove_flagged_days()
    print("Removed %d days (non wearing)." % n_removed_days)
    n_users = va.remove_wearables_without_valid_days()
    print("Removed %d wearables." % n_users)

    sbd = SleepBoudaryDetector(exp)

    sbd.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hrfullday",
                                hr_quantile=hr_quantile,
                                hr_volarity_threshold=hr_volarity,
                                hr_volatility_window_in_minutes=10,
                                hr_rolling_win_in_minutes=5,
                                hr_sleep_search_window=(start_hour, end_hour),
                                hr_min_window_length_in_minutes=hr_min_window_length,
                                hr_merge_blocks_gap_time_in_min=hr_merge_blocks,
                                hr_sleep_only_in_sleep_search_window=True,
                                hr_only_largest_sleep_period=True,
                                )

    sbd.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hrnight",
                                hr_quantile=hr_quantile,
                                hr_volarity_threshold=hr_volarity,
                                hr_volatility_window_in_minutes=10,
                                hr_rolling_win_in_minutes=5,
                                hr_sleep_search_window=(start_night, end_night),
                                hr_min_window_length_in_minutes=hr_min_window_length,
                                hr_merge_blocks_gap_time_in_min=hr_merge_blocks,
                                hr_sleep_only_in_sleep_search_window=True,
                                hr_only_largest_sleep_period=True,
                                )

    sbd.detect_sleep_boundaries(strategy="adapted_van_hees",
                                output_col="hyp_sleep_period_vanheesndw",
                                angle_cols=["pitch_mean_ndw", "roll_mean_ndw"],
                                angle_start_hour=start_hour,
                                angle_quantile=0.1,
                                angle_minimum_len_in_minutes=30,
                                angle_merge_tolerance_in_minutes=60,
                                angle_only_largest_sleep_period=True,  # This was missing
                                )

    sbd.detect_sleep_boundaries(strategy="adapted_van_hees",
                                output_col="hyp_sleep_period_vanheesdw",
                                angle_cols=["pitch_mean_dw", "roll_mean_dw"],
                                angle_start_hour=start_hour,
                                angle_quantile=0.1,
                                angle_minimum_len_in_minutes=30,
                                angle_merge_tolerance_in_minutes=60,
                                angle_only_largest_sleep_period=True,  # This was missing
                                )

    sbd.detect_sleep_boundaries(strategy="adapted_van_hees",
                                output_col="hyp_sleep_period_vanheesthigh",
                                angle_cols=["pitch_mean_thigh", "roll_mean_thigh"],
                                angle_start_hour=start_hour,
                                angle_quantile=0.1,
                                angle_minimum_len_in_minutes=30,
                                angle_merge_tolerance_in_minutes=60,
                                angle_only_largest_sleep_period=True,
                                )


    df_acc = []
    mses = {}
    cohens = {}

    print("Calculating evaluation measures...")
    for w in exp.get_all_wearables():

        sleep = {}
        sleep["diary"] = w.data[w.diary_sleep].astype(int)
        sleep["hrfullday"] = w.data["hyp_sleep_period_hrfullday"].astype(int)
        sleep["hrnight"] = w.data["hyp_sleep_period_hrnight"].astype(int)
        sleep["vanheesdw"] = w.data["hyp_sleep_period_vanheesdw"].astype(int)
        sleep["vanheesndw"] = w.data["hyp_sleep_period_vanheesndw"].astype(int)
        sleep["vanheesthigh"] = w.data["hyp_sleep_period_vanheesthigh"].astype(int)

        if sleep["diary"].shape[0] == 0:
            continue

        for comb in ["diary_hrfullday",
                     "diary_hrnight",
                     "diary_vanheesndw", "diary_vanheesdw",
                     "diary_vanheesthigh"
                     ]:
            a, b = comb.split("_")
            mses[comb] = mean_squared_error(sleep[a], sleep[b])
            cohens[comb] = cohen_kappa_score(sleep[a], sleep[b])

        tst_diary = w.get_total_sleep_time_per_day(based_on_diary=True)
        tst_hrfullday = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_hrfullday")
        tst_hrnight = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_hrnight")
        tst_vanheesndw = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_vanheesndw")
        tst_vanheesdw = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_vanheesdw")
        tst_vanheesthigh = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_vanheesthigh")

        onset_diary = w.get_onset_sleep_time_per_day(based_on_diary=True)
        onset_diary.name = "onset_diary"
        onset_hrfullday = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_hrfullday")
        onset_hrfullday.name = "onset_hrfullday"
        onset_hrnight = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_hrnight")
        onset_hrnight.name = "onset_hrnight"
        onset_vanheesndw = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanheesndw")
        onset_vanheesndw.name = "onset_vanheesndw"
        onset_vanheesdw = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanheesdw")
        onset_vanheesdw.name = "onset_vanheesdw"
        onset_vanheesthigh = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanheesthigh")
        onset_vanheesthigh.name = "onset_vanheesthigh"

        offset_diary = w.get_offset_sleep_time_per_day(based_on_diary=True)
        offset_diary.name = "offset_diary"
        offset_hrfullday = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_hrfullday")
        offset_hrfullday.name = "offset_hrfullday"
        offset_hrnight = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_hrnight")
        offset_hrnight.name = "offset_hrnight"
        offset_vanheesndw = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanheesndw")
        offset_vanheesndw.name = "offset_vanheesndw"
        offset_vanheesdw = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanheesdw")
        offset_vanheesdw.name = "offset_vanheesdw"
        offset_vanheesthigh = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanheesthigh")
        offset_vanheesthigh.name = "offset_vanheesthigh"

        df_res = pd.concat((onset_diary, onset_hrfullday,
                            onset_hrnight, onset_vanheesndw, onset_vanheesdw,
                            onset_vanheesthigh,
                            offset_diary, offset_hrfullday,
                            offset_hrnight, offset_vanheesndw, offset_vanheesdw,
                            offset_vanheesthigh,
                            tst_diary, tst_hrfullday,
                            tst_hrnight, tst_vanheesndw, tst_vanheesdw,
                            tst_vanheesthigh
                            ), axis=1)

        df_res["pid"] = w.get_pid()
        for comb in ["diary_hrfullday",
                     "diary_hrnight",
                     "diary_vanheesndw", "diary_vanheesdw",
                     "diary_vanheesthigh"
                     ]:
            df_res["mse_" + comb] = mses[comb]
            df_res["cohens_" + comb] = cohens[comb]

        # View signals
        # w.view_signals(["sleep", "diary"], sleep_cols=["hyp_sleep_period_vanheesthigh"], others=["pitch_mean_thigh", "roll_mean_thigh"])
        # w.view_signals(["sleep", "diary"], sleep_cols=["hyp_sleep_period_vanheesndw", "hyp_sleep_period_vanheesthigh"],
        #                others=["pitch_mean_thigh", "roll_mean_thigh", "hyp_invalid"])

        # v = Viewer(w)
        # v.view_signals(["sleep",  "diary"],
        #                sleep_cols=["hyp_sleep_period_hrfullday", "hyp_sleep_period_hrnight",
        #                            "hyp_sleep_period_vanheesndw", "hyp_sleep_period_vanheesdw",
        #                            "hyp_sleep_period_vanheesthigh"],
        #                colors=["green", "black", "blue", "orange", "yellow", "pink", "purple"],
        #                #alphas={'sleep': 0.3}
        #                )

        df_acc.append(df_res)

    exp_id += 1
    df_acc = pd.concat(df_acc)
    df_acc["exp_id"] = exp_id
    df_acc["quantile"] = hr_quantile
    df_acc["window_lengths"] = hr_min_window_length
    df_acc["time_merge_blocks"] = hr_merge_blocks

    df_acc.to_csv(os.path.join(output_path, "exp_%d.csv" % (exp_id)), index=False)

if __name__ == "__main__":

    diary_path = "../data/diaries/BBVS_new_diary.csv"  # <- TODO: CHANGE HERE
    data_path = "../data/small_collection_bvs/problem/BVS*.csv"  # <- TODO: CHANGE HERE

    hr_volarity = 5
    exp_id = 0

    quantiles = [0.325, 0.800]  # np.arange(0.1, 0.96, 0.025)
    window_lengths = [20]  # np.arange(10, 121, 5)
    time_merge_blocks = [90, 420]  # [-1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420]

    start_hour = 15
    end_hour = 15
    start_night = 23
    end_night = 7

    hparam_list = []
    exp_id = 0
    for hr_quantile in quantiles:
        for hr_merge_blocks in time_merge_blocks:
            for hr_min_window_length in window_lengths:

                exp_id += 1
                hparam_list.append([exp_id, data_path, diary_path, start_hour, end_hour, hr_quantile, hr_merge_blocks,
                                    hr_min_window_length, hr_volarity, start_night, end_night])

    print("Parameters: %d" % (len(hparam_list)))
    with tempfile.TemporaryDirectory() as output_path:

        futures = [one_loop.remote(hparam_list[i] + [output_path]) for i in range(len(hparam_list[:]))]
        print(ray.get(futures))
        #one_loop(hparam_list[0] + [output_path])

        # Cleaning up: Merge all experiments and
        dfs = glob(output_path + "/*")
        bigdf = pd.concat([pd.read_csv(f) for f in dfs])
        output_filename = "results_apr21_BBVS.csv.gz"
        bigdf.to_csv(output_filename, index=False)
        print("File: %s created!" % (output_filename))

    print("DONE!")

