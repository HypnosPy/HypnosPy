from glob import glob
from hypnospy import Wearable
from hypnospy.data import RawProcessing
from hypnospy.analysis import TimeSeriesProcessing
from hypnospy.analysis import PhysicalActivity
from hypnospy import Experiment


if __name__ == "__main__":

    # Configure an Experiment
    exp = Experiment()

    file_path = "./data/collection_mesa_hr/*1766*"

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
        w.configure_experiment_day(0)
        exp.add_wearable(w)

    tsp = TimeSeriesProcessing(exp)
    print("Valid days:", tsp.get_valid_days())
    print("Invalid days:", tsp.get_invalid_days())

    tsp.detect_sleep_boundaries(strategy="hr")
    #exp.get_wearable("1766").view_signals(["activity", "interval"])
    print("DONE")


