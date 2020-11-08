from unittest import TestCase

from hypnospy.data import RawProcessing
from hypnospy import Wearable
from hypnospy.analysis import TimeSeriesProcessing


class TestWearable(TestCase):

    def setUp(self):
        # Loads some file
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

    def test_get_activity_col(self):
        col_name = self.w_5day_invalid5hours.get_activity_col()
        self.assertEqual(col_name, "hyp_act_x")
        self.assertIsInstance(col_name, str)

    def test_get_pid(self):
        pid = self.w_5day_invalid5hours.get_pid()
        self.assertEqual(pid, "1")
        self.assertIsInstance(pid, str)

    def test_get_experiment_day_col(self):
        exp_day_col = self.w_5day_invalid5hours.get_experiment_day_col()
        self.assertEqual(exp_day_col, "hyp_exp_day")
        self.assertIsInstance(exp_day_col, str)

    def test_get_time_col(self):
        time_col = self.w_5day_invalid5hours.get_time_col()
        self.assertEqual(time_col, "hyp_time")
        self.assertIsInstance(time_col, str)

    def test_get_frequency_in_secs(self):
        freq = self.w_5day_invalid5hours.get_frequency_in_secs()
        self.assertEqual(freq, 30)
        self.assertIsInstance(freq, int)

    def test_get_epochs_in_min(self):
        nepochs = self.w_5day_invalid5hours.get_epochs_in_min()
        self.assertEqual(nepochs, 2.0)
        self.assertIsInstance(nepochs, float)

    def test_get_epochs_in_hour(self):
        nepochs = self.w_5day_invalid5hours.get_epochs_in_hour()
        self.assertEqual(nepochs, 120.0)
        self.assertIsInstance(nepochs, float)

    def test_change_start_hour_for_experiment_day(self):
        self.w_5day_invalid5hours.change_start_hour_for_experiment_day(0)

        # We are expecting to have only one experiment day and this will be day 5
        self.assertEqual(self.w_5day_invalid5hours.data["hyp_exp_day"].unique()[0], 5)

        # We now start our day at hour 18
        self.w_5day_invalid5hours.change_start_hour_for_experiment_day(18)
        # print(tsp.wearable.data[2155:2165])
        # Check if transition from artificial day 4 to day 5 is done correctly
        self.assertEqual(self.w_5day_invalid5hours.data.iloc[2159]["hyp_exp_day"], 4)
        self.assertEqual(self.w_5day_invalid5hours.data.iloc[2160]["hyp_exp_day"], 5)

        # Randomly change the start hour and return it back to 18
        self.w_5day_invalid5hours.change_start_hour_for_experiment_day(18)
        self.w_5day_invalid5hours.change_start_hour_for_experiment_day(0)
        self.w_5day_invalid5hours.change_start_hour_for_experiment_day(15)
        self.w_5day_invalid5hours.change_start_hour_for_experiment_day(18)
        self.assertEqual(self.w_5day_invalid5hours.data.iloc[2159]["hyp_exp_day"], 4)
        self.assertEqual(self.w_5day_invalid5hours.data.iloc[2160]["hyp_exp_day"], 5)

    def test_has_nan_activity(self):
        self.assertTrue(self.w_5day_invalid5hours.has_no_activity())
        self.w_5day_invalid5hours.fill_no_activity(1)
        self.assertFalse(self.w_5day_invalid5hours.has_no_activity())

    def test_valid_invalid_days(self):
        # Should not return anything yet, as we never marked any row as invalid
        invalid_days = self.w_5day_invalid5hours.get_invalid_days()
        self.assertSetEqual(invalid_days, set())

        valid_days = self.w_5day_invalid5hours.get_valid_days()
        self.assertSetEqual(valid_days, set([5]))

        #  We now force some an invalid day
        tsp = TimeSeriesProcessing(self.w_5day_invalid5hours)
        tsp.detect_non_wear(strategy="choi2011")
        tsp.check_valid_days(min_activity_threshold=0, max_non_wear_minutes_per_day=60)

        invalid_days = self.w_5day_invalid5hours.get_invalid_days()
        self.assertSetEqual(invalid_days, set({5}))
