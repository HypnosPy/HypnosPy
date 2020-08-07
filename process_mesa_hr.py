import tempfile
import pandas as pd
import os
from tqdm import tqdm
from glob import glob
from hypnospy import Wearable
from hypnospy.data import RawProcessing
from hypnospy.analysis import TimeSeriesProcessing
from hypnospy.analysis import PhysicalActivity
from hypnospy import Experiment
from hypnospy import Diary

from sklearn.metrics import mean_squared_error, cohen_kappa_score
from multiprocessing import Pool

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


def one_loop(hparam):

    exp_id, file_path, diary_path, start_hour, hr_quantile, hr_merge_blocks, hr_min_window_length = hparam

    exp = setup_experiment(file_path, diary_path, start_hour)

    print("Q:", hr_quantile, "W:", hr_min_window_length, "T", hr_merge_blocks)

    tsp = TimeSeriesProcessing(exp)

    tsp.fill_no_activity(-0.0001)
    tsp.detect_non_wear(strategy="none")
    tsp.check_valid_days(min_activity_threshold=0, max_non_wear_minutes_per_day=1000000, check_sleep_period=False,
                         check_diary=True)
    tsp.drop_invalid_days()

    tsp.detect_sleep_boundaries(strategy="annotation", output_col="hyp_sleep_period_psg",
                                annotation_merge_tolerance_in_minutes=300)

    tsp.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hr",
                                hr_quantile=hr_quantile,
                                hr_merge_blocks_delta_time_in_min=hr_merge_blocks,
                                hr_min_window_length_in_minutes=hr_min_window_length,
                                hr_volatility_window_in_minutes=10,
                                hr_volarity_threshold=5,
                                hr_rolling_win_in_minutes=5,
                                hr_sleep_search_window=(start_hour, start_hour - 1),
                                hr_sleep_only_in_sleep_search_window=True,
                                hr_only_largest_sleep_period=True,
                                )

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

        if w.get_pid() == "4590":
            print("USER 1")

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

    file_path = "./data/collection_mesa_hr_30_240/*"
    diary_path = "./data/diaries/mesa_diary.csv"
    start_hour = 15
    quantiles = [0.85] # [0.30, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]
    time_merge_blocks = [360] # [60, 120, 240, 300, 360]
    window_lengths = [45] #[27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50]

    hparam_list = []
    exp_id = 0
    for hr_quantile in quantiles:
        for hr_merge_blocks in time_merge_blocks:
            for hr_min_window_length in window_lengths:
                exp_id += 1
                hparam_list.append([exp_id, file_path, diary_path, start_hour, hr_quantile, hr_merge_blocks, hr_min_window_length])

    with tempfile.TemporaryDirectory() as output_path:
        #pool = Pool(processes=9)
        #pool.map(one_loop, hparam_list)
        one_loop(hparam_list[0])

        # Cleaning up: Merge all experiments and
        dfs = glob(output_path + "/*")
        bigdf = pd.concat([pd.read_csv(f) for f in dfs])
        bigdf.to_csv("results_MESA_hr_30_240_best.csv.gz", index=False)

    print("DONE")


