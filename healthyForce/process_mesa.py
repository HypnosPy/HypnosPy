from glob import glob
from hypnospy import Wearable, Experiment, Diary
from hypnospy.data import MESAPreProcessing
from hypnospy.analysis import NonWearingDetector, SleepBoudaryDetector, Viewer, PhysicalActivity, Validator, CircadianAnalysis
from hypnospy.analysis import SleepMetrics, SleepWakeAnalysis


def setup_experiment(file_path, diary_path, start_hour):
    # Configure an Experiment
    exp = Experiment()

    # Iterates over a set of files in a directory.
    # Unfortunately, we have to do it manually with RawProcessing because we are modifying the annotations
    for file in glob(file_path):
        pp = MESAPreProcessing(file)
        w = Wearable(pp)  # Creates a wearable from a pp object
        # Invert the two_stages flag. Now True means sleeping and False means awake

        w.data["interval_sleep"] = w.data["interval"].isin(["REST-S", "REST"])

        exp.add_wearable(w)
        exp.set_freq_in_secs(30)
        w.change_start_hour_for_experiment_day(start_hour)

    # diary = Diary().from_file(diary_path)
    # exp.add_diary(diary)

    return exp

if __name__ == "__main__":

    file_path = "../data/small_collection_mesa/*.csv"
    #file_path = "../data/collection_mesa_actigraphy/*.csv"
    diary_path = "../data/diaries/mesa_diary.csv"
    start_hour = 15
    end_hour = 15

    exp = setup_experiment(file_path, diary_path, start_hour)
    exp.fill_no_activity(-0.0001)
    exp.overall_stats()

    #
    # nwd = NonWearingDetector(exp)
    # nwd.detect_non_wear(strategy="choi", wearing_col="hyp_wearing_choi")
    #
    # # TODO: fix bug when annotation_merge_tolerance_in_minutes < 0
    sbd = SleepBoudaryDetector(exp)
    sbd.detect_sleep_boundaries(strategy="annotation", output_col="sleep_period_annotation",
                                annotation_col="interval_sleep",
                                annotation_merge_tolerance_in_minutes=30, annotation_only_largest_sleep_period=True)

    va = Validator(exp)
    va.flag_epoch_physical_activity_less_than(min_activity_threshold=0)
    va.flag_epoch_null_cols(col_list=["hyp_act_x"])

    nwd = NonWearingDetector(exp)
    nwd.detect_non_wear(strategy="choi", wearing_col="hyp_wearing_choi")
    va.flag_epoch_nonwearing("hyp_wearing_choi")

    va.flag_day_sleep_length_less_than(sleep_period_col="sleep_period_annotation", min_sleep_in_minutes=3*30)
    # n_removed_days = va.remove_flagged_days()
    # print("Removed %d days (short sleep)." % n_removed_days)

    va.flag_day_sleep_length_more_than(sleep_period_col="sleep_period_annotation", max_sleep_in_minutes=12*60)
    # n_removed_days = va.remove_flagged_days()
    # print("Removed %d days (long sleep)." % n_removed_days)

    va.flag_day_max_nonwearing(max_non_wear_minutes_per_day=3*10)
    # va.flag_day_max_nonwearing(max_non_wear_minutes_per_day=3*60)
    # n_removed_days = va.remove_flagged_days()
    # print("Removed %d days (non wearing)." % n_removed_days)

    va.flag_day_if_valid_epochs_smaller_than(valid_minutes_per_day=20*60)
    # n_removed_days = va.remove_flagged_days()
    # print("Removed %d days (few valid epochs)." % n_removed_days)

    va.validation_report()

    n_removed_wearables = va.remove_wearables_without_valid_days()
    print("Removed %d wearables." % n_removed_wearables)

    va.flag_day_if_not_enough_consecutive_days(3)
    n_removed_days = va.remove_flagged_days()
    print("Removed %d days that are not consecutive." % n_removed_days)
    n_removed_wearables = va.remove_wearables_without_valid_days()
    print("Removed %d wearables." % n_removed_wearables)

    exp.overall_stats()


    # Setting day to ml representation -> days may not be of fixed lengths.
    ml_column='ml_sequence'
    exp.set_ml_representation_days_exp(sleep_col="sleep_period_annotation", ml_column='ml_sequence')
    va.flag_day_sleep_length_less_than(sleep_period_col="sleep_period_annotation", min_sleep_in_minutes=3*30)    
    va.flag_day_sleep_length_more_than(sleep_period_col="sleep_period_annotation", max_sleep_in_minutes=12*60)
    n_removed_wearables = va.remove_wearables_without_valid_days()
    print("Removed %d wearables." % n_removed_wearables)

    # The below code takes time because they run linear algebra algorithms.
    ca = CircadianAnalysis(exp)
    ca.run_cosinor()
    ca.run_SSA()


    pa = PhysicalActivity(exp)
    pa.set_cutoffs(cutoffs=[58, 399, 1404], names=["sedentary", "light", "medium", "vigorous"])
    pa.generate_pa_columns(based_on="activity")
    bouts = pa.get_bouts("sedentary", length_in_minutes=30, sleep_col="sleep_period_annotation")

    sw = SleepWakeAnalysis(exp)
    sw.run_sleep_algorithm(algname="ScrippsClinic", activityIdx="hyp_act_x", rescoring=False, on_sleep_interval=False,
                           inplace=True)
    sw.run_sleep_algorithm(algname="Sadeh", activityIdx="hyp_act_x", rescoring=False, on_sleep_interval=False,
                           inplace=True)

    sm = SleepMetrics(exp)
    metrics = {}
    metrics["sleepEfficiency"] = sm.get_sleep_quality(sleep_metric="sleepEfficiency", wake_sleep_col="ScrippsClinic",
                                                      sleep_period_col="sleep_period_annotation")
    metrics["awakening"] = sm.get_sleep_quality(sleep_metric="awakening", wake_sleep_col="ScrippsClinic",
                                                sleep_period_col="sleep_period_annotation")
    metrics["arousal"] = sm.get_sleep_quality(sleep_metric="arousal", wake_sleep_col="ScrippsClinic",
                                              sleep_period_col="sleep_period_annotation")

    # SRI calculation will not work with set_ml_representation_days_exp because day representation will be of different lengths.
    # While SRI requires the days to be of fixed lengths.
    metrics["sri"] = sm.get_sleep_quality(sleep_metric="sri", wake_sleep_col="ScrippsClinic")

    r1 = sm.compare_sleep_metrics(ground_truth="ScrippsClinic", sleep_wake_col="Sadeh",
                             sleep_metrics=["sleepEfficiency", "awakenings"],
                             sleep_period_col="sleep_period_annotation", comparison_method="pearson")

    r2 = sm.compare_sleep_metrics(ground_truth="ScrippsClinic", sleep_wake_col="Sadeh",
                                  sleep_metrics=["sleepEfficiency", "awakenings"],
                                  sleep_period_col="sleep_period_annotation", comparison_method="relative_difference")

    sm.evaluate_sleep_metric(ground_truth="ScrippsClinic", sleep_wake_col="Sadeh", sleep_period_col="sleep_period_annotation")

    # View signals
    v = Viewer(exp)
    v.view_signals(["activity", "pa_intensity", "sleep"], sleep_cols=["sleep_period_annotation"],
                   signal_as_area=["ScrippsClinic"])


    print(metrics['sleepEfficiency'])
    
    w = exp.get_wearable("1768")
    print(w.cosinor.keys())
    print(w.ssa.keys())
    #w = exp.get_wearable("1766")


    #  ------------------------------------------------
    #
    # "XY.pickle"
    # X = {"demographics": [wearable list], {"mvpa_bouts": [wearable list], "hour_stats": [pid1day124, ]}
    # Y = {"sleep_efficiency": [0.5, 0.7...]}
    #
    #
    # DF is dataframe
    # pid, exp_day, exp_hour, mvpa_bouts, age, bmi, sleep_efficiency, ....
    # -
    # -
    # -
    #
    # collist = {"demographics": ["age", "bmi"], "PA": [A, B, C]}
