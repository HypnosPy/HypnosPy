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

if __name__ == "__main__":

    #parser = ArgumentParser()

    #parser.add_argument('--diary_path', type=str, default="./data/diaries/BBVS_new_diary.csv")
    #parser.add_argument('--data_path', type=str, default="./data/small_collection_bvs/BVS079C*.csv")

    #parser.add_argument('--hr_quantile', type=float, default=0.40)
    #parser.add_argument('--hr_merge_blocks', type=int, default=300)
    #parser.add_argument('--hr_min_window_length', type=int, default=40)
    #parser.add_argument('--hr_volarity', type=int, default=5)
    #args, unknown = parser.parse_known_args()

    #diary_path = "./data/diaries/BBVS_new_diary.csv"
    diary_path = "./data/diaries/NewBVSdiaries.csv"
    data_path = "./data/small_collection_bvs/DummyBVS*.csv"


    # parser.add_argument('--hr_quantile', type=float, default=0.40)
    # parser.add_argument('--hr_merge_blocks', type=int, default=300)
    # parser.add_argument('--hr_min_window_length', type=int, default=40)
    # parser.add_argument('--hr_volarity', type=int, default=5)
    hr_volarity = 5
    exp_id = 0
    quantiles = [0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525]
    time_merge_blocks = [60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    window_lengths = [25,27.5,30,32.5,35,37.5,40,42.5,45,47.5,50]

    with tempfile.TemporaryDirectory() as output_path:
        for hr_quantile in quantiles:
            for hr_merge_blocks in time_merge_blocks:
                for hr_min_window_length in window_lengths:

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
                        #pp.data["hyp_act_x"] = (pp.data["hyp_act_x"]/0.0060321) + 0.057

                        pp.data["hyp_act_x"] = pp.data["hyp_act_x"] - 1.0  # adjust for the BBVA dataset

                        # stdMET_highIC_Branch

                        w = Wearable(pp)  # Creates a wearable from a pp object
                        exp.add_wearable(w)

                    # Set frequency for every wearable in the collection
                    exp.set_freq_in_secs(60)

                    diary = Diary().from_file(diary_path)
                    exp.add_diary(diary)
                    exp.invalidate_days_without_diary()


                    tsp = TimeSeriesProcessing(exp)
                    tsp.fill_no_activity(-0.0001)
                    tsp.drop_invalid_days()

                    tsp.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hr",
                                                hr_quantile=hr_quantile,
                                                hr_volarity_threshold=hr_volarity,
                                                hr_volatility_window_in_minutes=10,
                                                hr_rolling_win_in_minutes=5,
                                                hr_sleep_search_window=(21, 11),
                                                hr_min_window_length_in_minutes=hr_min_window_length,
                                                hr_merge_blocks_delta_time_in_min=hr_merge_blocks,
                                                hr_sleep_only_in_sleep_search_window=True,
                                                hr_only_largest_sleep_period=True,
                                                )

                    pa = PhysicalActivity(exp, 1.5, 3, 6) # Should we use 1.5, 3 and 6 or 0.5, 2 and 5?
                    pa.generate_pa_columns()
                    mvpa_bouts = pa.get_mvpas(length_in_minutes=1, decomposite_bouts=False)
                    lpa_bouts = pa.get_lpas(length_in_minutes=1, decomposite_bouts=False)

                    df_acc = []

                    for w in exp.get_all_wearables():
                        print("Calculating evaluation measures...")
                        # w.view_signals(["activity", "hr", "pa_intensity", "sleep", "diary"])
                        diary_sleep = w.data[w.diary_sleep].astype(int)
                        hr_sleep = w.data["hyp_sleep_period_hr"].astype(int)
                        mse = mean_squared_error(diary_sleep, hr_sleep)
                        cohens_kappa = cohen_kappa_score(diary_sleep, hr_sleep)

                        print("Pid:", w.get_pid())
                        print("MSE: %.3f" % mse)
                        print("Cohen's Kappa: %.3f" % cohens_kappa)

                        w.change_start_hour_for_experiment_day(15)  # Experiment day now starts on 3pm
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

                        df_res = pd.concat((onset_hr, onset_diary, offset_hr, offset_diary, tst_diary, tst_hr), axis=1)

                        df_res["tst_diff"] = df_res["hyp_sleep_period_hr"] - df_res[w.diary_sleep]
                        df_res["tst_average"] = (df_res["hyp_sleep_period_hr"] + df_res[w.diary_sleep]) / 2.
                        df_res["pid"] = w.get_pid()
                        df_res["mse"] = mse
                        df_res["cohens"] = cohens_kappa

                        # View signals
                        # w.change_start_hour_for_experiment_day(0)
                        # w.view_signals(["activity", "hr", "pa_intensity", "sleep", "diary"], sleep_col="hyp_sleep_period_hr")

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
        bigdf.to_csv("send_to_joao.csv", index=False)

    print("DONE!")


