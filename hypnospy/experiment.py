from glob import glob

from hypnospy.data import RawProcessing
from hypnospy import Wearable


class Experiment(object):

    def __init__(self):
        self.datapath = None
        self.wearables = []

    def add_wearable(self, w):
        self.wearables.append(w)

    def configure_experiment(self,
                             datapath:str,
                             device_location:str,
                             # Configuration for activity
                             cols_for_activity,
                             is_emno:bool=False,
                             is_act_count:bool=False,
                             # Datatime parameters
                             col_for_datatime:str="time",
                             start_of_week:int=-1,
                             strftime:str=None,
                             # PID parameters
                             col_for_pid:str=None,
                             pid:int =-1,
                             additional_data:object=None
    ):

        # TODO: Missing a check to see if datapath exists.
        for file in glob(datapath + "/*"):
            pp = RawProcessing()
            pp.load_file(file,
                         device_location=device_location,
                         # activitiy information
                         cols_for_activity=cols_for_activity,
                         is_act_count=is_act_count,
                         is_emno=is_emno,
                         # Datatime information
                         col_for_datatime=col_for_datatime,
                         start_of_week=start_of_week,
                         strftime=strftime,
                         # Participant information
                         col_for_pid=col_for_pid,
                         pid=pid,
                         additional_data=additional_data
            )

            self.wearables.append(Wearable(pp))


