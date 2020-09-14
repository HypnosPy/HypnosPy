import pandas as pd
import numpy as np


class RawProcessing(object):

    def __init__(self,
                 filename: str,
                 # Configuration for activity
                 cols_for_activity,
                 col_for_mets: object = None,
                 is_emno: bool = False,
                 is_act_count: bool = False,
                 # Datetime parameters
                 col_for_datetime: str = "time",
                 start_of_week: int = -1,
                 strftime: str = None,
                 # PID parameters
                 col_for_pid: str = None,
                 pid: int = -1,
                 # HR parameters
                 col_for_hr: str = None,
                 # Any additional data?
                 additional_data: object = None,
                 device_location: str = None,
                 ):

        """
        :param filename: input filepath
        :param device_location: where this device was located (options are: "bw", "hip", "dw", "ndw", "chest", "hp_ch", "hp_bw", "all")
        :param additional_data:
        """
        # self.possible_locations = ["bw", "hip", "dw", "ndw", "chest", "hp_ch", "hp_bw", "all"]
        self.device = None
        self.filename = filename
        self.device_location = device_location
        self.additional_data = additional_data

        self.internal_activity_cols = ["hyp_act_x", "hyp_act_y", "hyp_act_z"]
        self.internal_time_col = "hyp_time_col"
        self.internal_mets_col = None
        self.naxis = 0
        self.is_act_count = False
        self.is_emno = False
        self.pid = None

        self.data = self.__load_wearable_data(self.filename)
        self.__configure_activity(cols_for_activity, col_for_mets, is_emno, is_act_count)
        self.__configure_datetime(col_for_datetime, strftime, start_of_week)
        self.__configure_pid(col_for_pid, pid)
        self.__configure_hr(col_for_hr)

        # if device_location not in self.possible_locations:
        #    print("ERROR: Device location '%s' not implemented. Options are %s" % (device_location, ','.join(self.possible_locations)))

    def get_pid(self):
        return self.pid

    def set_time_col(self, new_name: str):
        if new_name is not None:
            self.internal_time_col = new_name

    def __configure_hr(self, col_for_hr):
        self.internal_hr_col = col_for_hr

    def __configure_activity(self, cols_for_activity, col_for_mets, is_emno, is_act_count):
        self.is_act_count = is_act_count
        self.is_emno = is_emno
        self.naxis = len(cols_for_activity)

        if self.naxis == 0:
            raise ValueError("Need at least one col to represent activity.")

        if len(cols_for_activity) > 3:
            raise ValueError("Current implementation allows up to 3 cols for physical activity.")

        if col_for_mets is not None:
            self.internal_mets_col = col_for_mets

        for i, col in enumerate(cols_for_activity):
            if col not in self.data.keys():
                raise ValueError(
                    "Col %s not detected in the dataset. Possibilities are %s" % (col, ','.join(self.data.keys())))
            # If col exists, we save it with our internal name.
            # Note that in case only one axis is available, it will be 'hyp_act_x'.
            self.data[self.internal_activity_cols[i]] = self.data[col]

    def __configure_pid(self, col_for_pid: str, pid: int):
        if col_for_pid is None and pid == -1:
            raise ValueError("Either pid or col_for_pid need to have a valid value.")

        if pid != -1:
            self.pid = pid

        elif col_for_pid is not None:
            if col_for_pid not in self.data.keys():
                raise ValueError("Column %s is not in the dataframe." % (col_for_pid))

            pid = self.data.iloc[0][col_for_pid]
            self.pid = pid

    def __configure_datetime(self, col_for_datetime, strftime, start_of_week):
        if strftime is None and start_of_week is None:
            raise ValueError("Either strftime or start_of_week need to have a valid value.")

        # We need to figure out when the data started being collected.
        # Some datasets like MESA and HCHS from sleepdata.org do not have date information, unfortunately
        if strftime is None or (strftime is not None and strftime.find("%d") == -1):  # Could not find a day
            # Check if we can extract the start_of_week:
            starting_day_of_week = 1
            if type(start_of_week) == str:  # ColName from which we will extract the start_of_week
                if start_of_week not in self.data:
                    raise ValueError("%s is not a column in the dataframe" % (start_of_week))

                starting_day_of_week = self.data.iloc[0][start_of_week]

            elif type(start_of_week) == int:
                starting_day_of_week = start_of_week

            self.__datetime_without_date(col_for_datetime, starting_day_of_week)

        # We know when the week started because we have strftime well defined:
        else:
            self.data[self.internal_time_col] = pd.to_datetime(self.data[col_for_datetime], format=strftime)

    def __datetime_without_date(self, col_for_datatime, starting_day_of_week):
        """
        If we blindly use something like

            df["linetime"] = pd.to_datetime(df["linetime"])

        we will have all the days in the dataset set as the current day.
        That is not what we want.

        Instead, we pick a starting day (January 1, 2017 - *Sunday*) and modify the
        rest of the data according to the day of th week the experiment started.

        This way, if an experiment started on a Tuesday (day=3), the actigraphy
        data for this person would start on January 3, 2017.

        :param col_for_datatime: col name in the dataframe for the datetime
        :param starting_day_of_week: 0 = Sunday, 1 = Monday, ... 6 = Saturday
        """

        # TODO: this procedure to find the freq might be too specific and work only for mesa and latinos
        freq = abs(int(self.data[col_for_datatime].iloc[1][-2:]) - int(self.data[col_for_datatime].iloc[0][-2:]))
        ndays = int(np.ceil(self.data.shape[0] / (24 * (60 / freq) * 60)))
        firstTime, lastTime = self.data.iloc[0][col_for_datatime], self.data.iloc[-1][col_for_datatime]

        for n in range(-1, 2):
            times = pd.date_range(start="1-%d-2017 %s" % (starting_day_of_week, firstTime),
                                  end="1-%d-2017 %s" % (starting_day_of_week + ndays + n, lastTime),
                                  freq="{}s".format(freq))

            if times.shape[0] == self.data.shape[0]:
                self.data[self.internal_time_col] = times
                break
        else:
            raise ValueError("Could not find correct range for dataframe. "
                             "Please check if parameter ``datatime_col'' (=%s) is correct and has all its entries valid." % col_for_datatime)

    def export_hypnospy(self, filename):
        """

        :param filename:
        :return:
        """

        # TODO: find a better way to save these fields to file
        self.data.to_hdf(filename, key='data', mode='w')
        s = pd.Series([self.pid, self.internal_time_col,
                       self.internal_activity_cols[0:self.naxis],
                       self.internal_mets_col,
                       self.is_act_count, self.is_emno,
                       self.device_location, self.additional_data])
        s.to_hdf(filename, key='other')

        # mydict = dict(data=self.data,
        #               location=self.device_location,
        #               additional=self.additional_data)
        #
        # with h5py.File(filename, 'w') as hf:
        #     grp = hf.create_group('alist')
        #     for k, v in mydict.items():
        #         grp.create_dataset(k, data=v)

        # hf.create_dataset('data', data=mydata)

        # hf.create_dataset('location', data=)
        # hf.create_dataset('additional_data', data=self.additional_data)

        print("Saved file %s." % (filename))

    def __load_wearable_data(self, filename):
        """ Obtain device type
        Used to decide which way to parse the data from input file

        :return: Wearable type (Axivity, GeneActiv, Actigraph, Actiwatch and Apple Watch currently supported)
        """
        f = filename.lower()
        if f.endswith('.cwa') or f.endswith('.cwa.gz') or f.endswith('CWA'):
            return self.__process_axivity(filename)
        elif f.endswith('.bin'):
            return self.__process_geneactiv(filename)
        elif f.endswith('.dat'):
            # here ask for which device: ={ "applewatch":"Apple Watch","apple" [...] "actiwatch"[...]
            # ask if the data is raw or not--> if not--> move down the pecking order
            return self.__process_actigraph(filename)
        elif f.endswith('.csv') or f.endswith('.csv.gz'):
            return self.__process_csv(filename)
        else:
            print("ERROR: Wearable format not supported for file: " + filename)

    def __process_csv(self, csvfile):
        return pd.read_csv(csvfile)

    def __process_axivity(self, cwaFile):
        """ Use PAMPRO here
        """
        pass

    def __process_actigraph(self, datFile):
        """ Use PAMPRO here
        """
        pass

    def __process_geneactiv(self, datFile):
        """ Use PAMPRO here
        """
        pass

    def run_nonwear():
        """ Use PAMPRO autocalibration
        """
        pass

    def calibrate_data():
        """ Use PAMPRO autocalibration
        """
        pass

    def obtain_PA_metrics():
        """ Use PAMPRO for derivation of VM, VMHPF, ENMO, Pitch, Roll, Yaw
              Include proxy for MVPA, PA
        """
        # run this if class isn't count based
        # returns something that can be then summarized
        pass


class ActiwatchSleepData(RawProcessing):

    def __init__(self, filename, device_location=None, col_for_datetime="time", col_for_pid="pid"):
        super().__init__(filename, device_location=device_location,
                         cols_for_activity=["activity"],
                         is_act_count=True,
                         # Datatime information
                         col_for_datetime=col_for_datetime,
                         start_of_week="dayofweek",
                         # Participant information
                         col_for_pid=col_for_pid,
                         )
        self.device = "actigraphy"
        self.data["hyp_annotation"] = self.data["interval"].isin(["REST", "REST-S"])


# TODO: missing Actiheart (Fenland, BBVS), Axivity (BBVS, Biobank)

class MESAPreProcessing(RawProcessing):

    def __init__(self, filename, device_location=None, col_for_datetime="linetime", col_for_pid="mesaid"):
        super().__init__(filename, device_location=device_location,
                         cols_for_activity=["activity"],
                         is_act_count=True,
                         # Datatime information
                         col_for_datetime=col_for_datetime,
                         start_of_week="dayofweek",
                         # Participant information
                         col_for_pid=col_for_pid,
                         )
        self.device = "actigraphy"
        self.data["hyp_annotation"] = self.data["interval"].isin(["REST", "REST-S"])


class MMASHPreProcessing(RawProcessing):

    def __init__(self, filename, device_location=None, col_for_datetime="time", col_for_pid="pid",
                 col_for_hr="HR", cols_for_activity=["Axis1", "Axis2", "Axis3"], strftime="%Y-%b-%d %H:%M:%S"):
        super().__init__(filename, device_location=device_location,
                         # Activity information
                         cols_for_activity=cols_for_activity,
                         # Datatime information
                         col_for_datetime=col_for_datetime,
                         strftime=strftime,
                         # Participant information
                         col_for_pid=col_for_pid,
                         # HR information:
                         col_for_hr=col_for_hr,
                         )
        self.device = "actigraphy"
