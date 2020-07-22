from glob import glob
import pandas as pd

from hypnospy import Wearable
from hypnospy.data import RawProcessing
from hypnospy.analysis import TimeSeriesProcessing
from hypnospy.analysis import PhysicalActivity
from hypnospy import Experiment
from hypnospy import Diary

from sklearn.metrics import mean_squared_error, cohen_kappa_score

if __name__ == "__main__":

    # Configure an Experiment
    exp = Experiment()

    #diary_path = "./data/diaries/NewBBVSdiaries_werror.csv"
    diary_path = "./data/diaries/NewBVSdiaries.csv"
    data_path = "./data/small_collection_bvs/DummyBVS*.csv"

    # Iterates over a set of files in a directory.
    # Unfortunately, we have to do it manually with RawProcessing because we are modifying the annotations
    for file in glob(data_path):
        pp = RawProcessing(file,
                     # HR information
                     col_for_hr="mean_hr",
                     # Activity information
                     cols_for_activity=["ACC"],
                     is_act_count=False,
                     device_location="dw",
                     # Datetime information
                     col_for_datetime="REALTIME",
                     strftime="%d-%b-%Y %H:%M:%S",
                     # Participant information
                     col_for_pid="id")
        pp.data["hyp_act_x"] = (pp.data["hyp_act_x"]/0.0060321) + 0.057

        w = Wearable(pp)  # Creates a wearable from a pp object
        exp.add_wearable(w)

    diary = Diary().from_file(diary_path)
    exp.add_diary(diary)
    exp.invalidate_days_without_diary()

    tsp = TimeSeriesProcessing(exp)
    tsp.fill_no_activity(-0.0001)
    tsp.detect_sleep_boundaries(strategy="hr", output_col="hyp_sleep_period_hr", hr_quantile=0.425,
                                hr_volarity_threshold=5, hr_rolling_win_in_minutes=5, hr_sleep_search_window=(20, 12),
                                hr_min_window_length_in_minutes=40, hr_volatility_window_in_minutes=10,
                                hr_merge_blocks_delta_time_in_min=300, hr_sleep_only_in_sleep_search_window=True,
                                )

    pa = PhysicalActivity(exp, 1.5, 3, 6) # Should we use 1.5, 3 and 6 or 0.5, 2 and 5?
    pa.generate_pa_columns()
    mvpa_bouts = pa.get_mvpas(length_in_minutes=1, decomposite_bouts=False)
    lpa_bouts = pa.get_lpas(length_in_minutes=1, decomposite_bouts=False)

    df_tst_acc = []

    for w in exp.get_all_wearables():
        print("Calculating evaluation measures...")
        # w.view_signals(["activity", "hr", "pa_intensity", "sleep", "diary"])
        diary_sleep = w.data[w.diary_sleep].astype(int)
        hr_sleep = w.data["hyp_sleep_period_hr"].astype(int)
        print("Pid:", w.get_pid())
        print("MSE: %.3f" % mean_squared_error(diary_sleep, hr_sleep))
        print("Cohen's Kappa: %.3f" % cohen_kappa_score(diary_sleep, hr_sleep))

        w.change_start_hour_for_experiment_day(15)  # Experiment day now starts on 3pm
        tst_diary = w.get_total_sleep_time_per_day(based_on_diary=True)
        tst_hr = w.get_total_sleep_time_per_day(sleep_col="hyp_sleep_period_hr")

        df_tst = pd.merge(tst_diary, tst_hr, left_index=True, right_index=True)
        df_tst["diff"] = df_tst["hyp_sleep_period_hr"] - df_tst[w.diary_sleep]
        df_tst["average"] = (df_tst["hyp_sleep_period_hr"] + df_tst[w.diary_sleep]) / 2.
        df_tst["pid"] = w.get_pid()

        # View signals
        w.view_signals(["activity", "hr", "pa_intensity", "sleep", "diary"], sleep_col="hyp_sleep_period_hr")

        df_tst_acc.append(df_tst)

    df_tst_acc = pd.concat(df_tst_acc)
    df_tst_acc.to_csv("tst_results.csv", index=False)
    print("DONE!")
