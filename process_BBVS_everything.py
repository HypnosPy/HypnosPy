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

if __name__ == "__main__":

    diary_path = "./data/diaries/BBVS_new_diary.csv"
    data_path = "./data/small_collection_bvs/BVS*.csv"

    hr_volarity = 5
    exp_id = 0
    # full day HR
    fd_hr_quantile = 0.50
    fd_hr_window_length = 50
    fd_hr_time_merge_block = 60
    fd_hr_start_hour = 15
    fd_hr_end_hour = 15

    # night only HR
    no_hr_quantile = 0.50
    no_hr_window_length = 50
    no_hr_time_merge_block = 60
    no_hr_start_hour = 21
    no_hr_end_hour = 11

    with tempfile.TemporaryDirectory() as output_path:

        exp = load_experiment(data_path, diary_path, 15)

        tsp = TimeSeriesProcessing(exp)
        tsp.fill_no_activity(-0.0001)
        tsp.detect_non_wear(strategy="none")
        tsp.check_valid_days(min_activity_threshold=0, max_non_wear_minutes_per_day=180,
                             check_sleep_period=False)
        tsp.drop_invalid_days()

        tsp.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hr_fullday",
                                    hr_quantile=fd_hr_quantile,
                                    hr_volarity_threshold=hr_volarity,
                                    hr_volatility_window_in_minutes=10,
                                    hr_rolling_win_in_minutes=5,
                                    hr_sleep_search_window=(fd_hr_start_hour, fd_hr_end_hour),
                                    hr_min_window_length_in_minutes=fd_hr_window_length,
                                    hr_merge_blocks_delta_time_in_min=fd_hr_time_merge_block,
                                    hr_sleep_only_in_sleep_search_window=True,
                                    hr_only_largest_sleep_period=True,
                                    )

        tsp.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hr_night",
                                    hr_quantile=no_hr_quantile,
                                    hr_volarity_threshold=hr_volarity,
                                    hr_volatility_window_in_minutes=10,
                                    hr_rolling_win_in_minutes=5,
                                    hr_sleep_search_window=(no_hr_start_hour, no_hr_end_hour),
                                    hr_min_window_length_in_minutes=no_hr_window_length,
                                    hr_merge_blocks_delta_time_in_min=no_hr_time_merge_block,
                                    hr_sleep_only_in_sleep_search_window=True,
                                    hr_only_largest_sleep_period=True,
                                    )

        df_acc = []
        mses = {}
        cohens = {}

        print("Calculating evaluation measures...")
        for w in exp.get_all_wearables():

            sleep = {}
            sleep["diary"] = w.data[w.diary_sleep].astype(int)
            sleep["hrfullday"] = w.data["hyp_sleep_period_hr_fullday"].astype(int)
            sleep["hrnight"] = w.data["hyp_sleep_period_hr_night"].astype(int)
            #sleep["vanhees"] = w.data["hyp_sleep_period_vanhees"].astype(int)

            if sleep["diary"].shape[0] == 0:
                continue

            for comb in ["diary_hrfullday", "diary_hrnight"]: # , "diary_vanhees"]:
                a, b = comb.split("_")
                mses[comb] = mean_squared_error(sleep[a], sleep[b])
                cohens[comb] = cohen_kappa_score(sleep[a], sleep[b])

            tst_diary = w.get_total_sleep_time_per_day(based_on_diary=True)
            tst_hrfullday = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_hr_fullday")
            tst_hrnight = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_hr_night")

            onset_diary = w.get_onset_sleep_time_per_day(based_on_diary=True)
            onset_diary.name = "onset_diary"
            onset_hrfullday = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_hr_fullday")
            onset_hrfullday.name = "onset_hrfullday"
            onset_hrnight = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_hr_night")
            onset_hrnight.name = "onset_hrnight"

            offset_diary = w.get_offset_sleep_time_per_day(based_on_diary=True)
            offset_diary.name = "offset_diary"
            offset_hrfullday = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_hr_fullday")
            offset_hrfullday.name = "offset_hrfullday"
            offset_hrnight = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_hr_night")
            offset_hrnight.name = "offset_hrnight"

            df_res = pd.concat((onset_diary, onset_hrfullday, onset_hrnight,
                                offset_diary, offset_hrfullday, offset_hrnight,
                                tst_diary, tst_hrfullday, tst_hrnight), axis=1)

            df_res["pid"] = w.get_pid()
            for comb in ["diary_hrfullday", "diary_hrnight"]: #, "diary_vanhees"]:
                df_res["mse_" + comb] = mses[comb]
                df_res["cohens_" + comb] = cohens[comb]

            # View signals
            # w.change_start_hour_for_experiment_day(0)
            # w.view_signals(["activity", "hr", "pa_intensity", "sleep", "diary"], sleep_cols=["hyp_sleep_period_hr"])

            df_acc.append(df_res)

        exp_id += 1
        df_acc = pd.concat(df_acc)
        df_acc["exp_id"] = exp_id
        # TODO: need to upate these parameters
        df_acc["quantile"] = fd_hr_quantile
        df_acc["window_lengths"] = fd_hr_window_length
        df_acc["time_merge_blocks"] = fd_hr_time_merge_block

        df_acc.to_csv(os.path.join(output_path, "exp_%d.csv" % (exp_id)), index=False)

        # Cleaning up: Merge all experiments and
        dfs = glob(output_path + "/*")
        bigdf = pd.concat([pd.read_csv(f) for f in dfs])
        bigdf.to_csv("results_BBVS_everything.csv.gz", index=False)

    print("DONE!")


