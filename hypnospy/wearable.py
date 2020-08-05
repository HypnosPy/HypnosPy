from datetime import timedelta

import calendar

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
        self.diary_event = "hyp_diary_event"
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
        self.data[self.experiment_day_col] = (
                self.data[self.time_col] - pd.DateOffset(hours=hour_start_experiment)
        ).apply(lambda x: x.toordinal() - day_zero)

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
        self.data = self.data[self.data[self.experiment_day_col].isin(valid_days)]

    def add_diary(self, d: Diary):
        d.data = d.data[d.data["pid"] == self.get_pid()]
        self.diary = d
        self.data[self.diary_event] = False
        self.data.loc[self.data[self.time_col].isin(self.diary.data["sleep_onset"]), self.diary_event] = True
        self.data.loc[self.data[self.time_col].isin(self.diary.data["sleep_offset"]), self.diary_event] = True

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
            event = self.data[self.data[self.diary_event] == True]
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
            event = self.data[self.data[self.diary_event] == True]
        else:
            if sleep_col not in self.data.keys():
                raise ValueError("Could not find sleep_col (%s). Aborting." % sleep_col)
            event = self.data[self.data[sleep_col] == True]

        return event.groupby(self.experiment_day_col)[self.time_col].last()

    def view_signals(self, signals: list = ["activity", "hr", "pa_intensity", "sleep"],
                     others: list = [],
                     frequency: str = "60S",
                     sleep_cols: str = None):
        # TODO: finish implementing it!
        # TODO: probably we should move it to another viz package.
        import matplotlib.pyplot as plt
        import matplotlib.dates as dates
        cols = []

        for signal in signals:
            if signal == "activity":
                cols.append(self.get_activity_col())

            elif signal == "hr":
                if self.get_hr_col():
                    cols.append(self.get_hr_col())
                else:
                    raise KeyError("HR is not available for PID %s" % self.get_pid())

            elif signal == "pa_intensity":
                if hasattr(self, 'pa_intensity_cols'):
                    for pa in self.pa_intensity_cols:
                        if pa in self.data.keys():
                            cols.append(pa)

            elif signal == "sleep":
                for sleep_col in sleep_cols:
                    if sleep_col not in self.data.keys():
                        raise ValueError("Could not find sleep_col (%s). Aborting." % sleep_col)
                    cols.append(sleep_col)

            elif signal == "diary" and self.diary_event in self.data.keys():
                cols.append(self.diary_event)

            else:
                cols.append(signal)

        if len(cols) == 0:
            raise ValueError("Aborting: Empty list of signals to show.")

        cols.append(self.time_col)
        cols.append(self.experiment_day_col)
        for col in others:
            cols.append(col)

        df_plot = self.data[cols].set_index(self.time_col).resample(frequency).sum()

        # Daily version
        # dfs_per_day = [pd.DataFrame(group[1]) for group in df_plot.groupby(df_plot.index.day)]
        # Based on the experiment day gives us the correct chronological order of the days
        dfs_per_day = [pd.DataFrame(group[1]) for group in df_plot.groupby(self.experiment_day_col)]

        fig, ax1 = plt.subplots(len(dfs_per_day), 1, figsize=(14, 8))

        if len(dfs_per_day) == 1:
            ax1 = [ax1]

        plt.rcParams['font.size'] = 18
        plt.rcParams['image.cmap'] = 'plasma'
        plt.rcParams['axes.linewidth'] = 2
        plt.rc('font', family='serif')

        for idx in range(len(dfs_per_day)):
            maxy = 2

            # Resampling: hourly
            df2_h = dfs_per_day[idx]
            ax1[idx].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, rotation=0)
            ax1[idx].set_facecolor('snow')
            # ax1[idx].set_ylim(0, max(df_plot[self.get_activity_col()] / 10))
            ax1[idx].set_xlim(
                dfs_per_day[idx].index[0] - timedelta(hours=dfs_per_day[idx].index[0].hour - self.hour_start_experiment,
                                                      minutes=dfs_per_day[idx].index[0].minute,
                                                      seconds=dfs_per_day[idx].index[0].second),
                dfs_per_day[idx].index[0] - timedelta(hours=dfs_per_day[idx].index[0].hour - self.hour_start_experiment,
                                                      minutes=dfs_per_day[idx].index[0].minute,
                                                      seconds=dfs_per_day[idx].index[0].second) + timedelta(minutes=1439))

            if "activity" in signals:
                maxy = max(maxy, df2_h[self.get_activity_col()].max())
                ax1[idx].plot(df2_h.index, df2_h[self.get_activity_col()], label='Activity', linewidth=1,
                              color='black', alpha=1)

            if "pa_intensity" in signals:
                ax1[idx].fill_between(df2_h.index, 0, maxy, where=df2_h['hyp_vpa'], facecolor='forestgreen', alpha=1.0,
                                      label='VPA', edgecolor='forestgreen')
                only_mvpa = (df2_h['hyp_mvpa']) & (~df2_h['hyp_vpa'])
                ax1[idx].fill_between(df2_h.index, 0, maxy, where=only_mvpa, facecolor='palegreen', alpha=1.0,
                                      label='MVPA', edgecolor='palegreen')
                only_lpa = (df2_h['hyp_lpa']) & (~df2_h['hyp_mvpa']) & (~df2_h['hyp_vpa'])
                ax1[idx].fill_between(df2_h.index, 0, maxy, where=only_lpa, facecolor='honeydew', alpha=1.0,
                                      label='LPA', edgecolor='honeydew')
                ax1[idx].fill_between(df2_h.index, 0, maxy, where=df2_h['hyp_sed'], facecolor='palegoldenrod',
                                      alpha=0.6,
                                      label='sedentary', edgecolor='palegoldenrod')

            if "sleep" in signals:
                facecolors = ['royalblue', 'green', 'orange']
                maxy = 10000
                endy = 0
                addition = (maxy / len(facecolors))
                for i, sleep_col in enumerate(sleep_cols):
                    starty = endy
                    endy = endy + addition
                    sleeping = df2_h[sleep_col]  # TODO: get a method instead of an attribute
                    ax1[idx].fill_between(df2_h.index, starty, endy, where=sleeping, facecolor=facecolors[i],
                                          alpha=1, label=sleep_col)
                    # ax1[idx].fill_between(df2_h.index, 0, (df2_h['wake_window_0.4']) * 200, facecolor='cyan', alpha=1,
                    #                   label='wake')

            if "diary" in signals and self.diary_event in df2_h.keys():
                diary_event = df2_h[df2_h[self.diary_event] == True].index
                ax1[idx].vlines(x=diary_event, ymin=0, ymax=10000, facecolor='black', alpha=1, label='Diary',
                                linestyles="dashed")

            for col in others:
                ax1[idx].plot(df2_h.index, df2_h[col], label=col, linewidth=1, color='black', alpha=1)


            ax1[idx].set_ylabel("%d-%s\n%s" % (dfs_per_day[idx].index[-1].day,
                                               calendar.month_name[dfs_per_day[idx].index[-1].month][:3],
                                               calendar.day_name[dfs_per_day[idx].index[-1].dayofweek]),
                                rotation=0, horizontalalignment="right", verticalalignment="center")

            #ax1[idx].set_xticks([])
            ax1[idx].set_yticks([])
            #ax1[idx].set_label(["Experiment Day %d" % (idx)])

            # create a twin of the axis that shares the x-axis
            if "hr" in signals:
                ax2 = ax1[idx].twinx()
                # ax2.set_ylabel('mean_HR')  # we already handled the x-label with ax1
                ax2.plot(df2_h.index, df2_h[self.get_hr_col()], label='HR', color='red')
                ax2.set_ylim(df2_h[self.get_hr_col()].min() - 5, df2_h[self.get_hr_col()].max() + 5)
                ax2.set_xticks([])
                ax2.set_yticks([])

        ax1[0].set_title("%s" % self.get_pid(), fontsize=16)
        ax1[-1].set_xlabel('Time')
        ax1[-1].xaxis.set_minor_locator(dates.HourLocator(interval=4))  # every 4 hours
        ax1[-1].xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))  # hours and minutes

        handles, labels = ax1[-1].get_legend_handles_labels()
        # handles2, labels2 = ax2.get_legend_handles_labels()
        # fig.legend(handles + handles2, labels + labels2, loc='lower center', ncol=4)
        # return fig
        # ax.figure.savefig('%s_signals.pdf' % (self.get_pid()))
        # fig.suptitle("%s" % self.get_pid(), fontsize=16)

        fig.legend(handles, labels, loc='lower center', ncol=len(cols), fontsize=12)
        fig.savefig('%s_signals.pdf' % (self.get_pid()))
