import pandas as pd
import numpy as np

class RawProcessing(object):

    def __init__(self):

        """
        :param filename: input filepath
        :param device_location: where this device was located (options are: "bw", "hip", "dw", "ndw", "chest", "hp_ch", "hp_bw", "all")
        :param additional_data:
        """
        self.filename = None
        self.data = None
        self.device_location = None
        self.additional_data = None

        self.possible_locations = ["bw", "hip", "dw", "ndw", "chest", "hp_ch", "hp_bw", "all"]
        self.internal_time_col = "hyp_time"
        self.internal_activity_cols = ["hyp_act_x", "hyp_act_y", "hpy_act_z"]
        self.naxis = 0
        self.is_act_count = False
        self.is_emno = False

    def load_file(self,
                  filename:str,
                  device_location:str,
                  # Configuration for activity
                  cols_for_activity,
                  is_emno=False,
                  is_act_count=False,
                  # Datatime parameters
                  col_for_datatime:str="time",
                  start_of_week:int=-1,
                  strftime:str=None,
                  # PID parameters
                  col_for_pid:str=None,
                  pid:int=-1,
                  # Any additional data?
                  additional_data: object = None,
                  ):

        self.filename = filename
        if device_location not in self.possible_locations:
            print("ERROR: Device location '%s' not implemented. Options are %s" % (device_location, ','.join(self.possible_locations)))

        self.device_location = device_location
        self.additional_data = additional_data

        self.data = self.__load_wearable_data(self.filename)
        self.__configure_activity(cols_for_activity, is_emno, is_act_count)
        self.__configure_datatime(col_for_datatime, strftime, start_of_week)
        self.__configure_pid(col_for_pid, pid)


    def __configure_activity(self, cols_for_activity, is_emno, is_act_count):
        self.is_act_count = is_act_count
        self.is_emno = is_emno
        self.naxis = len(cols_for_activity)

        if self.naxis == 0:
            raise ValueError("Need at least one col to represent activity.")

        for i, col in enumerate(cols_for_activity):
            if col not in self.data.keys():
                raise ValueError("Col %s not detected in the dataset. Possibilities are %s" % (col, ','.join(self.data.keys())))
            # If col exists, we save it with our internal name.
            # Note that in case only one axis is available, it will be 'hyp_act_x'.
            self.data[self.internal_activity_cols[i]] = self.data[col]

    def __configure_pid(self, col_for_pid:str, pid:int):
        if col_for_pid is None and pid == -1:
            raise ValueError("Either pid or col_for_pid need to have a valid value.")

        if pid != -1:
            self.pid = pid

        elif col_for_pid is not None:
            if col_for_pid not in self.data.keys():
                raise ValueError("Column %s is not in the dataframe." % (col_for_pid))

            pid = self.data.iloc[0][col_for_pid]
            self.pid = pid

    def __configure_datatime(self, col_for_datatime, strftime, start_of_week):
        if strftime is None and start_of_week is None:
            raise ValueError("Either strftime or start_of_week need to have a valid value.")

        # We need to figure out when the data started being collected.
        # Some datasets like MESA and HCHS from sleepdata.org do not have date information, unfortunately
        if strftime is None or (strftime is not None and strftime.find("%d") == -1): # Could not find a day
            # Check if we can extract the start_of_week:
            starting_day_of_week = 1
            if type(start_of_week) == str:  # ColName from which we will extract the start_of_week
                if start_of_week not in self.data:
                    raise ValueError("%s is not a column in the dataframe" % (start_of_week))

                starting_day_of_week = self.data.iloc[0][start_of_week]

            elif type(start_of_week) == int:
                starting_day_of_week = start_of_week

            self.__datatime_without_date(col_for_datatime, starting_day_of_week)

        # We know when the week started because we have strftime well defined:
        else:
            self.data[self.internal_time_col] = pd.to_datatime(self.data[col_for_datatime], format=strftime)

    def __datatime_without_date(self, col_for_datatime, starting_day_of_week):
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
                             "Please check if parameter ``datatime_col'' is correct.")


    def export_hypnospy(self, filename):
        """

        :param filename:
        :return:
        """

        # TODO: find a better way to save these fields to file
        self.data.to_hdf(filename, key='data', mode='w')
        s = pd.Series([self.pid, self.internal_time_col,
                       self.internal_activity_cols[0:self.naxis],
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

        #hf.create_dataset('data', data=mydata)

        #hf.create_dataset('location', data=)
        #hf.create_dataset('additional_data', data=self.additional_data)

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
        elif f.endswith('.csv'):
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


    ### Activity Index (counts- from another repo- modify)
    def extract_activity_index(self):
        """
        Calculates the activity index feature on each 24 hour day.
        """
        try:
            os.mkdir(self.sub_dst + "/activity_index_days")  # set up output directory
        except OSError:
            pass

        count = 0

        # get days
        days = sorted(
            [
                self.sub_dst + "/raw_days/" + i
                for i in os.listdir(self.sub_dst + "/raw_days/")
                if ".DS_Store" not in i
            ]
        )
        for day in days:
            count += 1

            # load data
            df = pd.read_hdf(day)
            activity = []
            header = ["Time", "activity_index"]
            idx = 0
            window = int(self.window_size * self.fs)
            incrementer = int(self.window_size * self.fs)

            # iterate through windows
            while idx < len(df) - incrementer:
                # preprocessing: BP Filter
                temp = df[["X", "Y", "Z"]].iloc[idx : idx + window]
                start_time = temp.index[0]
                temp.index = range(len(temp.index))  # reset index
                temp = band_pass_filter(
                    temp, self.fs, bp_cutoff=self.band_pass_cutoff, order=3
                )

                # activity index extraction
                bp_channels = [
                    i for i in temp.columns.values[1:] if "bp" in i
                ]  # band pass filtered channels
                activity.append(
                    [
                        start_time,
                        activity_index(temp, channels=bp_channels).values[0][0],
                    ]
                )
                idx += incrementer

            # save data
            activity = pd.DataFrame(activity)
            activity.columns = header
            activity.set_index("Time", inplace=True)
            dst = "/activity_index_days/{}_activity_index_day_{}.h5".format(
                self.src_name, str(count).zfill(2)
            )
            activity.to_hdf(
                self.sub_dst + dst, key="activity_index_data_24hr", mode="w"
            )

    def find_sleepwin():
        # Apply our method based on HR here
        # if no HR look for expert annotations _anno
        # if no annotations look for diaries
        # if no diaries apply Van Hees heuristic method
        pass

    def obtain_data():
        """ Here we will define epochperiod, columns to be included, and generate
        the new .hpy file
        """
        pass


class DataReader():
    r""" Reads processed raw files and files obtained that are already proceessed"

    Params
    ---------

    """

    def __init__(self, wearable_type, datareader=[]):

        #store wearable type
        self.__wearable_type = wearable_type

        # store list of data reader
        self.__datareader = datareader

    @property
    def wearable_type(self):
        r""" This function will take in the type of reader"""
        #1. Check file type from above processing
        #2,.
        return

    @property
    def datareader(self):
        r""" This function will """
        #1. Check file type
        #2. Determine
        return


