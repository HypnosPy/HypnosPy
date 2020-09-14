import hypnospy
import pandas as pd
from hypnospy import Diary
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
        # Diary
        self.diary = None
        self.diary_onset = "hyp_diary_onset"
        self.diary_offset = "hyp_diary_offset"
        self.diary_sleep = "hyp_diary_sleep"

        if isinstance(input, str):
            # Reads a hypnosys file from disk
            self.__read_hypnospy(input)

        elif isinstance(input, hypnospy.data.preprocessing.RawProcessing):
            self.__read_preprocessing_obj(input)
            # print("Loaded wearable with pid %s" % (self.get_pid()))

        # Creates the experiment day and set it the initial hour to be midnight
        self.change_start_hour_for_experiment_day(0)

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

    def change_start_hour_for_experiment_day(self, hour_start_experiment):
        """
        Allows the experiment to start in another time than 00:00.

        :param hour_start_experiment: 0: midnight, 1: 01:00AM ...
        """
        self.hour_start_experiment = hour_start_experiment
        day_zero = self.data.iloc[0][self.time_col].toordinal()
        new_exp_day = (self.data[self.time_col] - pd.DateOffset(hours=hour_start_experiment)).apply(
            lambda x: x.toordinal() - day_zero)

        self.data[self.experiment_day_col] = new_exp_day

    def get_activity_col(self):
        return self.activitycols[0]

    def get_hr_col(self):
        return self.hr_col

    def get_invalid_days(self):
        """

        :return: list of invalid days in the dataset.
        """

        if self.experiment_day_col not in self.data.keys():
            # If it was not configured yet, we start the experiment day from midnight.
            self.change_start_hour_for_experiment_day(0)

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
        self.data = self.data[self.data[self.experiment_day_col].isin(valid_days)].copy()

    def add_diary(self, d: Diary):
        d.data = d.data[d.data["pid"] == self.get_pid()]
        self.diary = d
        self.data[self.diary_onset] = False
        self.data[self.diary_offset] = False
        self.data.loc[self.data[self.time_col].isin(self.diary.data["sleep_onset"]), self.diary_onset] = True
        self.data.loc[self.data[self.time_col].isin(self.diary.data["sleep_offset"]), self.diary_offset] = True

        self.data[self.diary_sleep] = False
        for _, row in self.diary.data.iterrows():
            if not pd.isna(row["sleep_onset"]) and not pd.isna(row["sleep_offset"]):
                self.data.loc[(self.data[self.time_col] >= row["sleep_onset"]) & (
                    self.data[self.time_col] <= row["sleep_offset"]), self.diary_sleep] = True

    def invalidate_days_without_diary(self):
        tst = self.get_total_sleep_time_per_day(based_on_diary=True)
        # Gets the experiment days with 0 total sleep time (i.e., no diary entry)
        invalid_days = set(tst[tst["hyp_diary_sleep"] == 0].index)
        # Flag them as invalid
        if len(invalid_days):
            self.data.loc[self.data[self.experiment_day_col].isin(invalid_days), self.invalid_col] = True

    def invalidate_all(self):
        self.data[self.invalid_col] = True

    def get_total_sleep_time_per_day(self, sleep_col: str = None, based_on_diary: bool = False):
        """

        :param sleep_col:
        :param based_on_diary:
        :return: A Series indexed by experiment_day with total of minutes slept per day
        """

        if not based_on_diary and sleep_col is None:
            raise ValueError("Unable to calculate total sleep time."
                             " You have to specify a sleep column or set ``based_on_diary`` to True "
                             "(assuming you previously added a diary.")
        if based_on_diary:
            if self.diary is None:
                raise ValueError("Diary not found for PID %s. Add a diary with ``add_diary``." % (self.get_pid()))
            return self.data.groupby(self.experiment_day_col)[[self.diary_sleep]].apply(
                lambda x: x.sum() / self.get_epochs_in_min())

        else:
            if sleep_col not in self.data.keys():
                raise ValueError("Could not find sleep_col named %s for PID %s. Aborting." % (self.get_pid(), sleep_col))
            return self.data.groupby(self.experiment_day_col)[[sleep_col]].apply(
                lambda x: x.sum() / self.get_epochs_in_min())

    def get_onset_sleep_time_per_day(self, sleep_col: str = None, based_on_diary: bool = False):
        """

        :param sleep_col:
        :param based_on_diary:
        :return: A Series indexed by experiment_day with total of minutes slept per day
        """

        if not based_on_diary and sleep_col is None:
            raise ValueError("Unable to calculate total sleep time."
                             " You have to specify a sleep column or set ``based_on_diary`` to True "
                             "(assuming you previously added a diary.")
        if based_on_diary:
            if self.diary is None:
                raise ValueError("Diary not found. Add a diary with ``add_diary``.")
            event = self.data[self.data[self.diary_onset] == True]
        else:
            if sleep_col not in self.data.keys():
                raise ValueError("Could not find sleep_col (%s). Aborting." % sleep_col)
            event = self.data[self.data[sleep_col] == True]

        return event.groupby(self.experiment_day_col)[self.time_col].first()

    def get_offset_sleep_time_per_day(self, sleep_col: str = None, based_on_diary: bool = False):
        """

        :param sleep_col:
        :param based_on_diary:
        :return: A Series indexed by experiment_day with total of minutes slept per day
        """

        if not based_on_diary and sleep_col is None:
            raise ValueError("Unable to calculate total sleep time."
                             " You have to specify a sleep column or set ``based_on_diary`` to True "
                             "(assuming you previously added a diary.")
        if based_on_diary:
            if self.diary is None:
                raise ValueError("Diary not found. Add a diary with ``add_diary``.")
            event = self.data[self.data[self.diary_offset] == True]
        else:
            if sleep_col not in self.data.keys():
                raise ValueError("Could not find sleep_col (%s). Aborting." % sleep_col)
            event = self.data[self.data[sleep_col] == True]

        return event.groupby(self.experiment_day_col)[self.time_col].last()


