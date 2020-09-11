from glob import glob
import os
import warnings

from hypnospy.data import RawProcessing
from hypnospy import Wearable
from hypnospy import Diary


class Experiment(object):

    def __init__(self):
        self.datapath = None
        self.wearables = {}

    def add_wearable(self, w):
        self.wearables[w.get_pid()] = w

    def configure_experiment(self, datapath: str, device_location: str, cols_for_activity, col_for_mets: str = None,
                             is_emno: bool = False, is_act_count: bool = False, col_for_datetime: str = "time",
                             start_of_week: int = -1, strftime: str = None, col_for_pid: str = None, pid: int = -1,
                             additional_data: object = None, col_for_hr: str = None):

        # TODO: Missing a check to see if datapath exists.
        if os.path.isdir(datapath):
            if not datapath.endswith("*"):
                datapath = os.path.join(datapath, "*")
        else:
            if '*' not in datapath:
                datapath = datapath + "*"

        for file in glob(datapath):
            pp = RawProcessing(file,
                               device_location=device_location,
                               # activitiy information
                               cols_for_activity=cols_for_activity,
                               is_act_count=is_act_count,
                               is_emno=is_emno,
                               # Datatime information
                               col_for_datetime=col_for_datetime,
                               col_for_mets=col_for_mets,
                               start_of_week=start_of_week,
                               strftime=strftime,
                               # Participant information
                               col_for_pid=col_for_pid,
                               pid=pid,
                               additional_data=additional_data,
                               # HR information
                               col_for_hr=col_for_hr, )
            w = Wearable(pp)
            self.wearables[w.get_pid()] = w

    def add_diary(self, diary: Diary):
        for pid in diary.data["pid"].unique():
            di = diary.data[diary.data["pid"] == pid]
            w = self.get_wearable(pid)
            if w:
                w.add_diary(Diary().from_dataframe(di))

    def get_wearable(self, pid: str):
        if pid in self.wearables.keys():
            return self.wearables[pid]
        else:
            warnings.warn("Unknown PID %s." % pid)
            return None
            # raise KeyError("Unknown PID %s." % pid)

    def size(self):
        return len(self.wearables)

    def get_all_wearables(self):
        return list(self.wearables.values())

    def set_freq_in_secs(self, freq: int):
        for wearable in self.get_all_wearables():
            wearable.set_frequency_in_secs(freq)

    def invalidate_days_without_diary(self):
        for wearable in self.get_all_wearables():
            wearable.invalidate_days_without_diary()

    def change_start_hour_for_experiment_day(self, hour):
        for wearable in self.get_all_wearables():
            wearable.change_start_hour_for_experiment_day(hour)
