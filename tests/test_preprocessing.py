from unittest import TestCase

from hypnospy.data import RawProcessing

class TestRawProcessing(TestCase):

    def test_load_file(self):

        pp1 = RawProcessing()
        pp1.load_file("../data/examples_mesa/mesa-sample.csv",
                      collection_name="mesa",
                      col_for_datatime="linetime",
                      device_location="dw",
                      start_of_week="dayofweek",
                      col_for_pid="mesaid")
        nrows = pp1.data.shape[0]
        self.assertEqual(nrows, 23003)
        self.assertIn("linetime", pp1.data.keys())

        pp2 = RawProcessing()
        pp2.load_file("../data/examples_mesa/mesa-sample.csv",
                      collection_name="mesa",
                      col_for_datatime="linetime",
                      device_location="dw",
                      start_of_week=4,
                      pid=1)

        nrows = pp2.data.shape[0]
        self.assertEqual(nrows, 23003)
        self.assertIn("hyp_time", pp2.data.keys())

    def test_export_hypnospy(self):
        pass

    def test_run_nonwear(self):
        #self.fail()
        pass

    def test_calibrate_data(self):
        pass
        #self.fail()

    def test_obtain_pa_metrics(self):
        pass

    def test_extract_activity_index(self):
        pass

    def test_find_sleepwin(self):
        pass

    def test_obtain_data(self):
        pass
