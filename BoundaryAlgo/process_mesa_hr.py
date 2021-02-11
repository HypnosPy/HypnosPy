import tempfile
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from glob import glob
from hypnospy import Wearable
from hypnospy.data import RawProcessing
from hypnospy.analysis import NonWearingDetector, SleepBoudaryDetector, Validator
from hypnospy import Experiment
from hypnospy import Diary

from sklearn.metrics import mean_squared_error, cohen_kappa_score
import ray

#ray.init()


def setup_experiment(file_path, diary_path, start_hour):
    # Configure an Experiment
    exp = Experiment()

    # Iterates over a set of files in a directory.
    # Unfortunately, we have to do it manually with RawProcessing because we are modifying the annotations
    for file in glob(file_path):
        pp = RawProcessing(file,
                           device_location="dw",
                           # HR information
                           col_for_hr="mean_hr",
                           # Activity information
                           cols_for_activity=["activity"],
                           is_act_count=True,
                           # Datetime information
                           col_for_datetime="linetime",
                           strftime="%Y-%m-%d %H:%M:%S",
                           # Participant information
                           col_for_pid="mesaid")

        w = Wearable(pp)  # Creates a wearable from a pp object
        # Invert the two_stages flag. Now True means sleeping and False means awake
        w.data["hyp_annotation"] = (w.data["stages"] > 0)
        exp.add_wearable(w)
        exp.set_freq_in_secs(30)
        w.change_start_hour_for_experiment_day(start_hour)

    diary = Diary().from_file(diary_path)
    exp.add_diary(diary)

    return exp

#@ray.remote
def one_loop(hparam):

    exp_id, file_path, diary_path, start_hour, end_hour, hr_quantile, hr_merge_blocks, hr_min_window_length, output_path = hparam

    exp = setup_experiment(file_path, diary_path, start_hour)
    exp.fill_no_activity(-0.0001)

    print("Q:", hr_quantile, "W:", hr_min_window_length, "T", hr_merge_blocks)

    va = Validator(exp)
    va.remove_wearables_without_diary()

    va.flag_epoch_physical_activity_less_than(min_activity_threshold=0)
    va.flag_epoch_null_cols(col_list=["hyp_act_x"])
    va.flag_day_max_nonwearing(max_non_wear_minutes_per_day=3*60)

    va.flag_day_if_invalid_epochs_larger_than(max_invalid_minutes_per_day=5 * 60)
    va.flag_day_without_diary()

    n_removed_days = va.remove_flagged_days()
    print("Removed %d days (non wearing)." % n_removed_days)
    n_users = va.remove_wearables_without_valid_days()
    print("Removed %d wearables." % n_users)

    sbd = SleepBoudaryDetector(exp)
    sbd.detect_sleep_boundaries(strategy="annotation", output_col="hyp_sleep_period_psg",
                                annotation_col="hyp_annotation",
                                annotation_merge_tolerance_in_minutes=300)

    sbd.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hr", hr_quantile=hr_quantile,
                                hr_volarity_threshold=5, hr_rolling_win_in_minutes=5,
                                hr_sleep_search_window=(start_hour, end_hour),
                                hr_min_window_length_in_minutes=hr_min_window_length,
                                hr_volatility_window_in_minutes=10, hr_merge_blocks_gap_time_in_min=hr_merge_blocks,
                                hr_sleep_only_in_sleep_search_window=True, hr_only_largest_sleep_period=True)

    df_acc = []
    mses = {}
    cohens = {}

    print("Calculating evaluation measures...")
    for w in tqdm(exp.get_all_wearables()):

        if w.data.empty:
            print("Data for PID %s is empty!" % w.get_pid())
            continue

        sleep = {}
        sleep["diary"] = w.data[w.diary_sleep].astype(int)
        sleep["hr"] = w.data["hyp_sleep_period_hr"].astype(int)
        sleep["psg"] = w.data["hyp_sleep_period_psg"].astype(int)

        for comb in ["diary_hr", "diary_psg", "psg_hr"]:
            a, b = comb.split("_")
            mses[comb] = mean_squared_error(sleep[a], sleep[b])
            cohens[comb] = cohen_kappa_score(sleep[a], sleep[b])

        tst_diary = w.get_total_sleep_time_per_day(based_on_diary=True)
        tst_psg = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_psg")
        tst_hr = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")

        onset_diary = w.get_onset_sleep_time_per_day(based_on_diary=True)
        onset_diary.name = "onset_diary"
        onset_psg = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_psg")
        onset_psg.name = "onset_psg"
        onset_hr = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")
        onset_hr.name = "onset_hr"

        offset_diary = w.get_offset_sleep_time_per_day(based_on_diary=True)
        offset_diary.name = "offset_diary"
        offset_psg = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_psg")
        offset_psg.name = "offset_psg"
        offset_hr = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")
        offset_hr.name = "offset_hr"

        df_res = pd.concat((onset_hr, onset_psg, onset_diary,
                            offset_hr, offset_psg, offset_diary,
                            tst_psg, tst_hr, tst_diary), axis=1)

        df_res["pid"] = w.get_pid()
        for comb in ["diary_hr", "diary_psg", "psg_hr"]:
            df_res["mse_" + comb] = mses[comb]
            df_res["cohens_" + comb] = cohens[comb]

        # View signals
        # w.view_signals(["sleep", "diary"],
        #                others=["hyp_annotation", "mean_hr"],
        #                sleep_cols=["hyp_sleep_period_psg", "hyp_sleep_period_hr"],
        #                frequency="30S"
        #                )

        df_acc.append(df_res)

    df_acc = pd.concat(df_acc)
    df_acc["exp_id"] = exp_id
    df_acc["quantile"] = hr_quantile
    df_acc["window_lengths"] = hr_min_window_length
    df_acc["time_merge_blocks"] = hr_merge_blocks

    df_acc.to_csv(os.path.join(output_path, "exp_%d.csv" % (exp_id)), index=False)


if __name__ == "__main__":

    file_path = "../data/collection_mesa_hr_30_240/*"
    diary_path = "../data/diaries/mesa_diary.csv"
    start_hour = 15
    end_hour = 15

    quantiles = np.arange(0.1, 0.96, 0.025)
    window_lengths = np.arange(10, 121, 5)
    time_merge_blocks = [-1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420]

    hparam_list = []
    exp_id = 0
    for hr_quantile in quantiles:
        for hr_merge_blocks in time_merge_blocks:
            for hr_min_window_length in window_lengths:
                exp_id += 1
                hparam_list.append([exp_id, file_path, diary_path, start_hour, end_hour, hr_quantile, hr_merge_blocks, hr_min_window_length])


    with tempfile.TemporaryDirectory() as output_path:

        # futures = [one_loop.remote(hparam_list[i] + [output_path]) for i in range(len(hparam_list[:]))]
        # print(ray.get(futures))
        one_loop(hparam_list[0] + [output_path])

        # Cleaning up: Merge all experiments and
        dfs = glob(output_path + "/*")
        bigdf = pd.concat([pd.read_csv(f) for f in dfs])
        bigdf.to_csv("results_jan21_MESA_hr.csv.gz", index=False)

    print("DONE")


