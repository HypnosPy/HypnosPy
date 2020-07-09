import hypnospy
import pandas as pd
import h5py


class Wearable(object):

    def __init__(self, input):
        """

        :param input: Either a path to a PreProcessing file saved with ``export_hyp`` or a PreProcessing object
        """
        self.data = None
        self.freq_in_secs = None
        self.hour_start_experiment = None
        self.experiment_day_col = "hyp_exp_day"
        self.invalid_col = "hyp_invalid"

        # Other fields
        self.mets_col = None
        self.device_location = None
        self.additional_data = None
        # Participant Info
        self.pid = None
        # Activity Info
        self.activitycols = None
        self.mets_col = None
        self.is_act_count = None
        self.is_emno = None
        # Time Info
        self.time_col = None
        # HR Info
        self.hr_col = None

        if type(input) == str:
            # Reads a hypnosys file from disk
            self.__read_hypnospy(input)

        elif type(input) == hypnospy.data.preprocessing.RawProcessing:
            self.__read_preprocessing_obj(input)
            print("Loaded wearable with pid %s" % (self.get_pid()))

    def __read_preprocessing_obj(self, input):
        "input is a wearable object. We can copy its fields"
        self.data = input.data
        self.device_location = input.device_location
        self.additional_data = input.additional_data
        # Participant Info
        self.pid = str(input.pid)
        # Activity Info
        self.activitycols = input.internal_activity_cols
        self.mets_col = input.internal_mets_col
        self.is_act_count = input.is_act_count
        self.is_emno = input.is_emno
        # Time Info
        self.time_col = input.internal_time_col
        # HR Info
        self.hr_col = input.internal_hr_col

    def __read_hypnospy(self, filename):

        self.data = pd.read_hdf(filename, 'data')
        l = pd.read_hdf(filename, 'other')
        self.pid, self.time_col, self.activitycols, self.internal_mets_col, self.is_act_count, self.is_emno, \
        self.device_location, self.additional_data = l

        self.pid = str(self.pid)

        # hf = h5py.File(filename, 'r')
        # self.data = hf.get('data')
        # self.device_location = hf.get('location')
        # self.additional_data = hf.get('additional_data')

    def get_pid(self):
        return self.pid

    def get_experiment_day_col(self):
        return self.experiment_day_col

    def get_mets_col(self):
        return self.mets_col

    def get_time_col(self):
        return self.time_col

    def set_frequency_in_secs(self, freq):
        self.freq_in_secs = freq

    def get_frequency_in_secs(self):
        if self.freq_in_secs:
            return self.freq_in_secs

        freq_str = pd.infer_freq(self.data[self.time_col])
        if freq_str is None:
            raise ValueError("Could not infer the frequency for pid %s." % self.get_pid())
        # pd.to_timedelta requires we have a number
        if not freq_str[0].isdigit():
            freq_str = "1" + freq_str
        return int(pd.to_timedelta(freq_str).total_seconds())

    def get_epochs_in_min(self):
        # TODO: should we force it to be integer?
        return 60 / self.get_frequency_in_secs()

    def get_epochs_in_hour(self):
        # TODO: should we force it to be integer?
        return 60 * self.get_epochs_in_min()

    def fill_no_activity(self, value):
        # TODO: write the use case for triaxial devices.
        self.data[self.get_activity_col()].fillna(value, inplace=True)

    def has_no_activity(self):
        return self.data[self.get_activity_col()].isnull().any()

    def configure_experiment_day(self, hour_start_experiment):
        """
        Allows the experiment to start in another time than 00:00.

        :param hour_start_experiment: 0: midnight, 1: 01:00AM ...
        """
        self.hour_start_experiment = hour_start_experiment
        self.data[self.experiment_day_col] = (
                self.data[self.time_col] - pd.DateOffset(hours=hour_start_experiment)).dt.day

    def get_activity_col(self):
        return self.activitycols[0]

    def get_invalid_days(self):
        """

        :return: list of invalid days in the dataset.
        """

        if self.experiment_day_col not in self.data.keys():
            # If it was not configured yet, we start the experiment day from midnight.
            self.configure_experiment_day(0)

        if self.invalid_col not in self.data.keys():
            self.data[self.invalid_col] = False

        grp_days = self.data.groupby([self.experiment_day_col])[self.invalid_col].any().reset_index()
        return set(grp_days[grp_days[self.invalid_col] == True][self.experiment_day_col].unique())

    def get_valid_days(self):
        """

        :return: list of valid days in the dataset.
        """
        invalid_days = self.get_invalid_days()
        all_days = set(self.data[self.experiment_day_col].unique())
        return all_days - invalid_days

    def drop_invalid_days(self):
        valid_days = self.get_valid_days()

        self.data = self.data[self.data[self.experiment_day_col].isin(valid_days)]
