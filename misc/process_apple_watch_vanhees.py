import tempfile
import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from hypnospy import Wearable
from hypnospy.data import RawProcessing
from hypnospy.analysis import TimeSeriesProcessing
from hypnospy import Experiment

from sklearn.metrics import mean_squared_error, cohen_kappa_score
from multiprocessing import Pool


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

def one_loop(hparam):

    exp_id, file_path, start_hour, end_hour, quantile, merge_blocks, min_window_length = hparam

    exp = load_experiment(file_path, start_hour)

    print("Q:", quantile, "W:", min_window_length, "T", merge_blocks)

    tsp = TimeSeriesProcessing(exp)

    tsp.fill_no_activity(-0.0001)
    tsp.detect_sleep_boundaries(strategy="annotation", output_col="hyp_sleep_period_psg",
                                annotation_merge_tolerance_in_minutes=300)

    tsp.detect_sleep_boundaries(strategy="adapted_van_hees", output_col="hyp_sleep_period_vanhees", angle_cols=[],
                                angle_use_triaxial_activity=True, angle_start_hour=start_hour, angle_quantile=quantile,
                                angle_minimum_len_in_minutes=min_window_length,
                                angle_merge_tolerance_in_minutes=merge_blocks)

    df_acc = []
    mses = {}
    cohens = {}

    print("Calculating evaluation measures...")
    for w in tqdm(exp.get_all_wearables()):

        sleep = {}
        sleep["psg"] = w.data["hyp_sleep_period_psg"].astype(int)
        sleep["vanhees"] = w.data["hyp_sleep_period_vanhees"].astype(int)


        for comb in ["psg_vanhees"]:
            a, b = comb.split("_")
            mses[comb] = mean_squared_error(sleep[a], sleep[b])
            cohens[comb] = cohen_kappa_score(sleep[a], sleep[b])

        tst_psg = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_psg")
        tst_vanhees = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_vanhees")

        onset_psg = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_psg")
        onset_psg.name = "onset_psg"
        onset_vanhees = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanhees")
        onset_vanhees.name = "onset_vanhees"

        offset_psg = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_psg")
        offset_psg.name = "offset_psg"
        offset_vanhees = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanhees")
        offset_vanhees.name = "offset_vanhees"

        df_res = pd.concat((onset_psg, onset_vanhees,
                            offset_psg, offset_vanhees,
                            tst_psg, tst_vanhees), axis=1)

        df_res["pid"] = w.get_pid()
        for comb in ["psg_vanhees"]:
            df_res["mse_" + comb] = mses[comb]
            df_res["cohens_" + comb] = cohens[comb]

        # View signals
        # w.view_signals(["sleep"],
        #                others=["hyp_annotation", "hr", "hyp_act_x", "hyp_act_y", "hyp_act_z"],
        #                sleep_cols=["hyp_sleep_period_psg", "hyp_sleep_period_hr", "hyp_sleep_period_vanhees"],
        #                frequency="30S"
        #                )

        df_acc.append(df_res)

    df_acc = pd.concat(df_acc)
    df_acc["exp_id"] = exp_id
    df_acc["quantile"] = quantile
    df_acc["window_lengths"] = min_window_length
    df_acc["time_merge_blocks"] = merge_blocks

    df_acc.to_csv(os.path.join(output_path, "exp_%d.csv" % (exp_id)), index=False)


if __name__ == "__main__":

    # Configure an Experiment
    exp = Experiment()

    data_path = "./data/collection_apple_watch/*.csv"
    start_hour = 15
    end_hour = start_hour - 1

    quantiles = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275,
                 0.30, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675,
                 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]
    window_lengths = [25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50]
    time_merge_blocks = [30, 60, 120, 240, 360, 420]

    hparam_list = []
    exp_id = 0
    for quantile in quantiles:
        for merge_blocks in time_merge_blocks:
            for min_window_length in window_lengths:
                exp_id += 1
                hparam_list.append([exp_id, data_path, start_hour, end_hour, quantile, merge_blocks, min_window_length])


    with tempfile.TemporaryDirectory() as output_path:
        pool = Pool(processes=8)
        pool.map(one_loop, hparam_list)

        #one_loop(hparam_list[0])

        # Cleaning up: Merge all experiments and
        dfs = glob(output_path + "/*")
        bigdf = pd.concat([pd.read_csv(f) for f in dfs])
        bigdf.to_csv("results_apple_vanhees.csv.gz", index=False)

    print("DONE")

