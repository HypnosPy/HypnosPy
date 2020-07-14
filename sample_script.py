from hypnospy import Wearable
from hypnospy.data import RawProcessing
from hypnospy.analysis import SleepWakeAnalysis
from hypnospy.analysis import TimeSeriesProcessing
from hypnospy.analysis import PhysicalActivity
from hypnospy import Experiment


if __name__ == "__main__":

    # Configure an Experiment
    exp = Experiment()

    # exp.configure_experiment(datapath="./data/small_collection_mesa/",
    #                          cols_for_activity=["activity"],
    #                          is_act_count=True,
    #                          # Datatime information
    #                          col_for_datatime="linetime",
    #                          device_location="dw",
    #                          start_of_week="dayofweek",
    #                          # Participant information
    #                          col_for_pid="mesaid"
    #                          )


    pp = RawProcessing()
    pp.load_file(#"./data/examples_mesa/mesa-sample.csv",
                 "./data/examples_mesa/mesa-sample-day5-invalid5hours.csv",
                 # activitiy information
                 cols_for_activity=["activity"],
                 is_act_count=True,
                 # Datatime information
                 col_for_datatime="linetime",
                 device_location="dw",
                 start_of_week="dayofweek",
                 # Participant information
                 col_for_pid="mesaid")

    pp.data["hyp_annotation"] = pp.data["interval"].isin(["REST", "REST-S"])
    w1 = Wearable(pp)  # Creates a wearable from a pp object

    exp.add_wearable(w1)

    # User can either use an individual wearable or a list of them with experiments.
    #exp = Experiment(collection_name="mesa")
    #exp.add(w)

    # swa = SleepWakeAnalysis(w1)
    # r = swa.oakley_algorithm()

    tsp = TimeSeriesProcessing(w1)

    tsp.detect_non_wear(strategy="choi")

    #tsp.check_valid_days(max_non_wear_min_per_day=180, min_activity_threshold=0)
    #print(tsp.wearable.data["hyp_invalid"])
    #print("Valid days:", tsp.get_valid_days())
    #print("Invalid days:", tsp.get_invalid_days())

    #tsp.wearable.data.loc[tsp.wearable.data[tsp.experiment_day].isin({8,10}), tsp.invalid_col] = True
    #print("Valid days:", tsp.get_valid_days())

    #tsp.check_consecutive_days(5)
    #print("Valid days:", tsp.get_valid_days())

    #tsp.drop_invalid_days(inplace=True)
    #print("Valid days:", tsp.get_valid_days())
    #print("Invalid days:", tsp.get_invalid_days())

    #tsp.detect_sleep_boundaries(strategy="annotation", annotation_hour_to_start_search=18) # TODO: missing test.
    #print("Valid days:", tsp.get_valid_days())

    #tsp.check_valid_days(max_non_wear_min_per_day=180, min_activity_threshold=0)

    # TODO: test sleep_metrics
    #tsp.drop_invalid_days()

    # TODO: PA bouts? How to?
    #pa = PhysicalActivity(w1, 399, 1404)
    #mvpa_bouts = pa.get_mvpas(length_in_minutes=1, decomposite_bouts=False)
    #lpa_bouts = pa.get_lpas(length_in_minutes=1, decomposite_bouts=False)

    #pa_bins = pa.get_binned_pa_representation()
    #pa_stats = pa.get_stats_pa_representation()

    print("DONE")
    #tsp.cosinor()

    #ca = CosinorAnalysis(w1)
    #ca.get_cosine()

    #print(r)
    #print(w.data)

    #if trixial→ collapse ENMO to (15’’,30’’)
    #Determine sampling rate (15’’, 30’’, 1’) → if not, ERROR (‘Device sampling rate not supported’)
    #pp.export_hypnospy("dummy.hpy") # -> [ typeOfDevice (triaxial, hr, counts), typeOfStudy(full, night_only), location(dw,ndw,hip,chest,bw,bw_ch,bw_hp,hp_ch,all), additional(diary,anno,PSGlabel), df={ _pid, _time, _acc, _hr?, "PSGLabel"} ]
    #Pampro -> ourformat.hd5



