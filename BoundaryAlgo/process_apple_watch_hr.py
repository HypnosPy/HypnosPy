import tempfile
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from hypnospy import Wearable
from hypnospy.data import RawProcessing
from hypnospy import Experiment
from hypnospy.analysis import SleepBoudaryDetector, Viewer

from sklearn.metrics import mean_squared_error, cohen_kappa_score
import ray

ray.init()

def load_experiment(data_path, start_hour):
    # Configure an Experiment
    exp = Experiment()

    # Iterates over a set of files in a directory.
    # Unfortunately, we have to do it manually with RawProcessing because we are modifying the annotations
    for file in glob(data_path):
        pp = RawProcessing(file,
                     # HR information
                     col_for_hr="hr",
                     # Activity information
                     cols_for_activity=["x", "y", "z"],
                     is_act_count=True,
                     # Datetime information
                     col_for_datetime="faketime",
                     strftime="%Y-%m-%d %H:%M:%S",
                     # Participant information
                     col_for_pid="pid")

        w = Wearable(pp)  # Creates a wearable from a pp object
        w.data["hyp_annotation"] = (w.data["label"] > 0)
        exp.add_wearable(w)

    # Set frequency for every wearable in the collection
    exp.set_freq_in_secs(15)

    # Changing the hour the experiment starts from midnight (0) to 3pm (15)
    exp.change_start_hour_for_experiment_day(start_hour)

    return exp

@ray.remote
def one_loop(hparam):

    exp_id, file_path, start_hour, end_hour, hr_quantile, hr_merge_blocks, hr_min_window_length, output_path = hparam

    exp = load_experiment(file_path, start_hour)
    exp.fill_no_activity(-0.0001)

    print("Q:", hr_quantile, "L:", hr_min_window_length, "G", hr_merge_blocks)

    sbd = SleepBoudaryDetector(exp)

    sbd.detect_sleep_boundaries(strategy="annotation", annotation_col="hyp_annotation",
                                output_col="hyp_sleep_period_psg",
                                annotation_merge_tolerance_in_minutes=300)

    sbd.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hr", hr_quantile=hr_quantile,
                                hr_volarity_threshold=5, hr_rolling_win_in_minutes=5,
                                hr_sleep_search_window=(start_hour, end_hour),
                                hr_min_window_length_in_minutes=hr_min_window_length,
                                hr_volatility_window_in_minutes=10, hr_merge_blocks_gap_time_in_min=hr_merge_blocks,
                                hr_sleep_only_in_sleep_search_window=True, hr_only_largest_sleep_period=True)

    sbd.detect_sleep_boundaries(strategy="adapted_van_hees", output_col="hyp_sleep_period_vanhees", angle_cols=[],
                                angle_use_triaxial_activity=True, angle_start_hour=start_hour, angle_quantile=0.1,
                                angle_minimum_len_in_minutes=30, angle_merge_tolerance_in_minutes=60)

    df_acc = []
    mses = {}
    cohens = {}

    print("Calculating evaluation measures...")
    for w in tqdm(exp.get_all_wearables()):

        sleep = {}
        sleep["psg"] = w.data["hyp_sleep_period_psg"].astype(int)
        sleep["hr"] = w.data["hyp_sleep_period_hr"].astype(int)
        sleep["vanhees"] = w.data["hyp_sleep_period_vanhees"].astype(int)


        for comb in ["psg_hr", "psg_vanhees", "hr_vanhees"]:
            a, b = comb.split("_")
            mses[comb] = mean_squared_error(sleep[a], sleep[b])
            cohens[comb] = cohen_kappa_score(sleep[a], sleep[b])

        tst_psg = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_psg")
        tst_hr = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")
        tst_vanhees = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_vanhees")

        onset_psg = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_psg")
        onset_psg.name = "onset_psg"
        onset_hr = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")
        onset_hr.name = "onset_hr"
        onset_vanhees = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanhees")
        onset_vanhees.name = "onset_vanhees"

        offset_psg = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_psg")
        offset_psg.name = "offset_psg"
        offset_hr = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")
        offset_hr.name = "offset_hr"
        offset_vanhees = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanhees")
        offset_vanhees.name = "offset_vanhees"

        df_res = pd.concat((onset_hr, onset_psg, onset_vanhees,
                            offset_hr, offset_psg, offset_vanhees,
                            tst_psg, tst_hr, tst_vanhees), axis=1)

        df_res["pid"] = w.get_pid()
        for comb in ["psg_hr", "psg_vanhees", "hr_vanhees"]:
            df_res["mse_" + comb] = mses[comb]
            df_res["cohens_" + comb] = cohens[comb]

        df_acc.append(df_res)

    df_acc = pd.concat(df_acc)
    df_acc["exp_id"] = exp_id
    df_acc["quantile"] = hr_quantile
    df_acc["window_lengths"] = hr_min_window_length
    df_acc["time_merge_blocks"] = hr_merge_blocks

    df_acc.to_csv(os.path.join(output_path, "exp_%d.csv" % exp_id), index=False)

    # View signals
    # v = Viewer(exp.get_all_wearables()[0])
    # v.view_signals(["sleep"],
    #                other_signals=["hyp_annotation"],
    #                sleep_cols=["hyp_sleep_period_psg", "hyp_sleep_period_hr", "hyp_sleep_period_vanhees"],
    #                #alphas={'sleep': 0.3}
    #                )


if __name__ == "__main__":

    # Configure an Experiment
    exp = Experiment()

    data_path = "../data/collection_apple_watch/*.csv"
    start_hour = 15
    end_hour = start_hour

    quantiles = np.arange(0.1, 0.96, 0.025)
    window_lengths = np.arange(10, 121, 5)
    time_merge_blocks = [-1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420]

    hparam_list = []
    exp_id = 0
    for hr_quantile in quantiles:
        for hr_merge_blocks in time_merge_blocks:
            for hr_min_window_length in window_lengths:
                exp_id += 1
                hparam_list.append([exp_id, data_path, start_hour, end_hour, hr_quantile, hr_merge_blocks, hr_min_window_length])

    with tempfile.TemporaryDirectory() as output_path:

        futures = [one_loop.remote(hparam_list[i] + [output_path]) for i in range(len(hparam_list[:]))]
        print(ray.get(futures))

        # one_loop(hparam_list[0] + [output_path])

        # # Cleaning up: Merge all experiments and
        dfs = glob(output_path + "/*")
        bigdf = pd.concat([pd.read_csv(f) for f in dfs])
        bigdf.to_csv("results_jan21_apple_hr.csv.gz", index=False)

    print("DONE")

