from unittest import TestCase

from hypnospy.data import RawProcessing
from hypnospy import Wearable
from hypnospy.analysis import TimeSeriesProcessing


class TestTimeSeriesProcessing(TestCase):

    def setUp(self):

        pp1 = RawProcessing()
        pp1.load_file("../data/examples_mesa/mesa-sample-day5-invalid5hours.csv",
                     # activitiy information
                     cols_for_activity=["activity"],
                     is_act_count=True,
                     # Datatime information
                     col_for_datatime="linetime",
                     device_location="dw",
                     start_of_week="dayofweek",
                     # Participant information
                     col_for_pid="mesaid")
        self.w_5day_invalid5hours = Wearable(pp1)


    def test_featurize(self):
        pass

    def test_day_night_split(self):
        pass

    def test_detect_sleep_boundaries(self):
        pass

    def test_fill_no_activity(self):
        pass

    def test_detect_non_wear(self):
        pass

    def test_configure_experiment_day(self):
        pass

    def test_check_valid_days(self):
        pass
