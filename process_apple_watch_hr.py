from glob import glob
from hypnospy import Wearable
from hypnospy.data import RawProcessing
from hypnospy.analysis import SleepWakeAnalysis
from hypnospy.analysis import TimeSeriesProcessing
from hypnospy.analysis import PhysicalActivity
from hypnospy import Experiment


if __name__ == "__main__":

    # Configure an Experiment
    exp = Experiment()

    file_path = "./data/collection_apple_watch/*.csv"

    # Iterates over a set of files in a directory.
    # Unfortunately, we have to do it manually with RawProcessing because we are modifying the annotations
    for file in glob(file_path):
        pp = RawProcessing(file,
                     # HR information
                     col_for_hr="hr",
                     # Activity information
                     cols_for_activity=["counts"],
                     is_act_count=True,
                     device_location="dw",
                     # Datetime information
                     col_for_datetime="faketime",
                     strftime="%Y-%m-%d %H:%M:%S",
                     # Participant information
                     col_for_pid="pid")

        w = Wearable(pp)  # Creates a wearable from a pp object
        w.configure_experiment_day(0)
        exp.add_wearable(w)

    tsp = TimeSeriesProcessing(exp)
    print("Valid days:", tsp.get_valid_days())
    print("Invalid days:", tsp.get_invalid_days())

    tsp.detect_sleep_boundaries(strategy="hr")

    print("Valid days:", tsp.get_valid_days())

    print("DONE")

