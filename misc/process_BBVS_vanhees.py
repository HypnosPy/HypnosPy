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


def load_experiment(data_path, diary_path):
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


# +

if __name__ == "__main__":

    diary_path = "./data/diaries/BBVS_new_diary.csv"
    # diary_path = "./data/diaries/NewBVSdiaries.csv"
    # data_path = "./data/small_collection_bvs/DummyBVS5*.csv"
    data_path = "./data/small_collection_bvs/BVS*.csv"

    hr_volarity = 5
    exp_id = 0
    quantiles = [0.05] #, 0.10, 0.15, 0.20, 0.25]
    window_lengths = [25] #, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50]
    time_merge_blocks = [30] #, 60, 90, 120, 180]
    start_hour = 15

    with tempfile.TemporaryDirectory() as output_path:
        for quantile in quantiles:
            for merge_blocks in time_merge_blocks:
                for min_window_length in window_lengths:

                    exp = load_experiment(data_path, diary_path)

                    tsp = TimeSeriesProcessing(exp)
                    tsp.fill_no_activity(-0.0001)
                    tsp.detect_non_wear(strategy="none")
                    tsp.check_valid_days(min_activity_threshold=0, max_non_wear_minutes_per_day=180,
                                         check_sleep_period=False)
                    tsp.drop_invalid_days()
                    tsp.detect_sleep_boundaries(strategy="adapted_van_hees", output_col="hyp_sleep_period_vanhees",
                                                angle_cols=["pitch_mean_dw", "roll_mean_dw"],
                                                angle_start_hour=start_hour, angle_quantile=quantile,
                                                angle_minimum_len_in_minutes=min_window_length,
                                                angle_merge_tolerance_in_minutes=merge_blocks)

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
                        vanhees_sleep = w.data["hyp_sleep_period_vanhees"].astype(int)

                        mses["diary_vanhees"] = mean_squared_error(diary_sleep, vanhees_sleep)
                        cohens["diary_vanhees"] = cohen_kappa_score(diary_sleep, vanhees_sleep)

                        tst_diary = w.get_total_sleep_time_per_day(based_on_diary=True)
                        tst_vanhees = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_vanhees")

                        onset_diary = w.get_onset_sleep_time_per_day(based_on_diary=True)
                        onset_diary.name = "onset_diary"
                        onset_vanhees = w.get_onset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanhees")
                        onset_vanhees.name = "onset_vanhees"

                        offset_diary = w.get_offset_sleep_time_per_day(based_on_diary=True)
                        offset_diary.name = "offset_diary"
                        offset_vanhees = w.get_offset_sleep_time_per_day(sleep_col="hyp_sleep_period_vanhees")
                        offset_vanhees.name = "offset_vanhees"

                        df_res = pd.concat((onset_diary, onset_vanhees,
                                            offset_diary, offset_vanhees,
                                            tst_diary, tst_vanhees), axis=1)

                        df_res["pid"] = w.get_pid()
                        for comb in ["diary_vanhees"]:
                            df_res["mse_" + comb] = mses[comb]
                            df_res["cohens_" + comb] = cohens[comb]

                        # View signals
                        # w.change_start_hour_for_experiment_day(0)
                        # w.view_signals(["activity", "hr", "pa_intensity", "sleep", "diary"], sleep_col="hyp_sleep_period_hr")

                        df_acc.append(df_res)

                    exp_id += 1
                    df_acc = pd.concat(df_acc)
                    df_acc["exp_id"] = exp_id
                    df_acc["quantile"] = quantile
                    df_acc["window_lengths"] = min_window_length
                    df_acc["time_merge_blocks"] = merge_blocks

                    df_acc.to_csv(os.path.join(output_path, "exp_%d.csv" % (exp_id)), index=False)

        # Cleaning up: Merge all experiments and
        dfs = glob(output_path + "/*")
        bigdf = pd.concat([pd.read_csv(f) for f in dfs])
        bigdf.to_csv("results_BBVS_vanhees.csv.gz", index=False)

    print("DONE!")
# -


