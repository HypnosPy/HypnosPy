import os
import tempfile
from glob import glob
import pandas as pd

from hypnospy import Wearable
from hypnospy.data import RawProcessing
from hypnospy.analysis import TimeSeriesProcessing
from hypnospy.analysis import PhysicalActivity
from hypnospy import Experiment
from hypnospy import Diary

from argparse import ArgumentParser

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
        # pp.data["hyp_act_x"] = (pp.data["hyp_act_x"]/0.0060321) + 0.057

        pp.data["hyp_act_x"] = pp.data["hyp_act_x"] - 1.0  # adjust for the BBVA dataset

        # stdMET_highIC_Branch

        w = Wearable(pp)  # Creates a wearable from a pp object
        exp.add_wearable(w)

    # Set frequency for every wearable in the collection
    exp.set_freq_in_secs(60)

    # Changing the hour the experiment starts from midnight (0) to 3pm (15)
    exp.change_start_hour_for_experiment_day(start_hour)

    diary = Diary().from_file(diary_path)
    exp.add_diary(diary)

    return exp


if __name__ == "__main__":

    #data_path = "./data/small_collection_bvs/olds/Dummy*.csv"
    #diary_path = "./data/diaries/NewBVSdiaries.csv"

    diary_path = "./data/diaries/BBVS_new_diary.csv"
    data_path = "./data/small_collection_bvs/BVS*.csv"

    hr_volarity = 5
    exp_id = 0
    quantiles = [0.75] # , 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525]
    window_lengths = [5] # , 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50]
    time_merge_blocks = [60] # , 120, 180, 240, 300, 360]
    start_hour = 15
    end_hour = start_hour - 1

    with tempfile.TemporaryDirectory() as output_path:
        for hr_quantile in quantiles:
            for hr_merge_blocks in time_merge_blocks:
                for hr_min_window_length in window_lengths:

                    exp = load_experiment(data_path, diary_path, start_hour)

                    tsp = TimeSeriesProcessing(exp)
                    tsp.fill_no_activity(-0.0001)
                    tsp.detect_non_wear(strategy="none")
                    tsp.check_valid_days(min_activity_threshold=0, max_non_wear_min_per_day=180, check_sleep_period=False)
                    tsp.drop_invalid_days()

                    tsp.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hr",
                                                hr_quantile=hr_quantile,
                                                hr_volarity_threshold=hr_volarity,
                                                hr_volatility_window_in_minutes=10,
                                                hr_rolling_win_in_minutes=5,
                                                hr_sleep_search_window=(start_hour, end_hour),
                                                hr_min_window_length_in_minutes=hr_min_window_length,
                                                hr_merge_blocks_delta_time_in_min=hr_merge_blocks,
                                                hr_sleep_only_in_sleep_search_window=True,
                                                hr_only_largest_sleep_period=True,
                                                )

                    # Dont change the intevals below: we're using 1.5, 3 and 6.
                    # Removed the -1 when creating the wearable
                    pa = PhysicalActivity(exp, 1.5, 3, 6)
                    pa.generate_pa_columns()
                    mvpa_bouts = pa.get_mvpas(length_in_minutes=1, decomposite_bouts=False)
                    lpa_bouts = pa.get_lpas(length_in_minutes=1, decomposite_bouts=False)

                    df_acc = []
                    mses = {}
                    cohens = {}

                    print("Calculating evaluation measures...")
                    for w in exp.get_all_wearables():
                        # w.data = w.data[w.data["hyp_exp_day"].isin([5])]

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
                        w.view_signals(["activity", "hr", "pa_intensity", "sleep", "diary"], sleep_cols=["hyp_sleep_period_hr"])

                        df_acc.append(df_res)

                    exp_id += 1
                    df_acc = pd.concat(df_acc)
                    df_acc["exp_id"] = exp_id
                    df_acc["quantile"] = hr_quantile
                    df_acc["window_lengths"] = hr_min_window_length
                    df_acc["time_merge_blocks"] = hr_merge_blocks

                    df_acc.to_csv(os.path.join(output_path, "exp_%d.csv" % (exp_id)), index=False)

        # Cleaning up: Merge all experiments and
        dfs = glob(output_path + "/*")
        bigdf = pd.concat([pd.read_csv(f) for f in dfs])
        bigdf.to_csv("results_BBVS_hr.csv.gz", index=False)

    print("DONE!")


