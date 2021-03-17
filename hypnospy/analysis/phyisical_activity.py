from hypnospy import Wearable
from hypnospy import Experiment
from hypnospy import misc

import pandas as pd
import numpy as np


class PhysicalActivity(object):
    """
    
    Class used to analyse the device wearer's activity signals based on the Wearable.data df
    
    """
    def __init__(self, input: {Experiment, Wearable}, cutoffs: list = None, names: list = None):
        """

        :param input: Either an experiment or wearable object.
        :param cutoffs: List of cut-offs
        :param names: List of physical activity names associated with cut-offs.
        Note: it is expected that len(names) will be len(cutoffs) + 1
        """

        if type(input) is Wearable:
            self.wearables = [input]
        elif type(input) is Experiment:
            self.wearables = input.get_all_wearables()

        self.set_cutoffs(cutoffs, names)

    def set_cutoffs(self, cutoffs, names):
        """
        This method is used to define physical activity cut-offs and their respective names.

        The cut-off values here are tightly related to the wearable device used.
        We suggest the user to read the latest research on it.
        Vincent van Hees' GGIR has a summarized documentation on this topic: see https://cran.r-project.org/web/packages/GGIR/vignettes/GGIR.html#published-cut-points-and-how-to-use-them


        :param cutoffs: List of values
        :param names:  List of names. Expected to have one element more than ``cutoffs``.
        :return: None
        """
        if cutoffs is None and names is None:
            return

        # N cut-offs defined N+1 classes
        assert len(cutoffs) == len(names) - 1

        # Check if cutoffs are in increasing order
        assert sorted(cutoffs) == cutoffs

        self.cutoffs = cutoffs
        self.names = names

    def generate_pa_columns(self, based_on="activity"):
        """
        Sets two physical activity (pa) attributes for each wearable in a given experiment (self). 
        The two attributes are pa_cutoffs, and pa_names.
        pa_cutoffs is a list of numbers.
        pa_names is a list of names representing the numbers.

        :param based_on: Base column used to calculate the physical activity.
        :return: None
        """

        if self.cutoffs is None or self.names is None:
            raise AttributeError("Please use `set_cutoffs` before using this method.")

        for wearable in self.wearables:
            # first cut-off
            wearable.data[self.names[0]] = wearable.data[based_on] <= self.cutoffs[0]
            # Other cut-ffs
            for i in range(1, len(self.names)):
                wearable.data[self.names[i]] = wearable.data[based_on] > self.cutoffs[i - 1]

            wearable.pa_cutoffs = self.cutoffs
            wearable.pa_names = self.names

    def get_bouts(self, pa_col: str, length_in_minutes: int, pa_allowance_in_minutes: int, resolution: str,
                  sleep_col: object = None) -> pd.DataFrame:
        """
        Return the bouts for a given physical activity column (``pa_col``).
        One bout is counted when ``pa_col`` is True for more than ``length_in_minutes``.
        We allow up to ``pa_allowance_in_minutes`` minutes of physical activity below the minimal required for a pa level.
        If ``sleep_col`` is used, we do not count bouts when data[sleep_col] is True.
        ``resolution`` can currently be either "day" or "hour".

        :param pa_col:                   The name of the physical activity column in the dataframe.
        :param length_in_minutes:        The minimal length of the activity in minutes
        :param pa_allowance_in_minutes:  The maximum allowance of minutes in which a bout is still counted.
        :param resolution:               Either "day" or "hour". The resolution expected for output.
        :param sleep_col:                If a valid binary column, we ignore bouts that happened when the value of this col is True.
                                         Make sure to run SleepBoudaryDetector.detect_sleep_boundaries() first.
        :return:                         A dataframe counting the number of bouts for the given physical activity level
        """

        if pa_col not in self.names:
            raise ValueError("Unknown physical activity column %s. Please use ``set_cutoffs``.")

        returning_df = []
        for wearable in self.wearables:

            if sleep_col and (sleep_col not in wearable.data.keys()):
                raise ValueError(
                    "Could not find sleep_col named %s for PID %s. Aborting." % (sleep_col, wearable.get_pid())
                )
            df = wearable.data.copy()
            min_num_epochs = wearable.get_epochs_in_min() * length_in_minutes

            df["pa_len"], df["pa_grp"] = misc.get_consecutive_series(df, pa_col)
            # We admit up to allowance minutes
            df[pa_col], df["pa_len"], df["pa_grp"] = misc.merge_sequences_given_tolerance(df, "hyp_time_col", pa_col,
                                                                                          pa_allowance_in_minutes,
                                                                                          seq_id_col="pa_grp",
                                                                                          seq_length_col="pa_len")

            # calculate all possible bouts, either including the sleep period or not
            if sleep_col:
                bouts = df[(df[pa_col] == True) & (df["pa_len"] >= min_num_epochs) & (df[sleep_col] == False)]
            else:
                bouts = df[(df[pa_col] == True) & (df["pa_len"] >= min_num_epochs)]

            # drop_duplicates is used to get only the first occurrence of a bout sequence.
            bouts = bouts[
                ["hyp_time_col", wearable.get_experiment_day_col(), "pa_grp", "pa_len", pa_col]
            ].drop_duplicates(subset=["pa_grp"])

            if resolution == "day":
                tmp_df = bouts.groupby([wearable.get_experiment_day_col()])[pa_col].sum().reset_index()
            elif resolution == "hour":
                gbouts = bouts.set_index("hyp_time_col")
                tmp_df = gbouts.groupby([wearable.get_experiment_day_col(), gbouts.index.hour])[
                    pa_col].sum().reset_index()
            else:
                raise ValueError("The parameter 'resolution' can only be `day` or `hour`.")

            tmp_df["pid"] = wearable.get_pid()
            tmp_df["bout_length"] = length_in_minutes

            returning_df.append(tmp_df)

        returning_df = [x for x in returning_df if type(x) == pd.DataFrame]
        return pd.concat(returning_df).reset_index(drop=True)

    def get_binned_pa_representation(self) -> pd.DataFrame:
        """
        Counts the number of epochs for each physical activity levels (see PhysicalActivity.names) per hour of the day.
        PhysicalActivity.names and PhysicalActivity.cutoffs are set with ``PhysicalActivity.set_cutoffs``

        Note the difference between this method and get_bouts.
        While get_bouts counts the number of bouts at a given hour, binner_pa counts the number of minutes at a given PA level per hour.
        If the number of epochs is smaller than the minimal for a bout, get_bouts would not capture it, while binned_pa_representation would.

        :return: dataframe with pa counts binned per hour
        """
        rows = []
        for wearable in self.wearables:
            act_hour = \
                wearable.data.groupby(
                    [wearable.get_experiment_day_col(), wearable.data[wearable.get_time_col()].dt.hour])[
                    wearable.get_activity_col()]

            pid = wearable.get_pid()

            PAs = []
            # Special case of self.names[0]
            tmpdf = act_hour.apply(lambda x: (x <= self.cutoffs[0]).sum())
            tmpdf.name = self.names[0]
            PAs.append(tmpdf)

            for i in range(1, len(self.cutoffs)):
                tmpdf = act_hour.apply(lambda x: x.between(self.cutoffs[i - 1], self.cutoffs[i]).sum())
                tmpdf.name = self.names[i]
                PAs.append(tmpdf)

            # Special case for the last activity
            tmpdf = act_hour.apply(lambda x: (x >= self.cutoffs[-1]).sum())
            tmpdf.name = self.names[-1]
            PAs.append(tmpdf)

            concatenated = pd.concat(PAs, axis=1)
            concatenated["pid"] = pid
            rows.append(concatenated)

        return pd.concat(rows)

    def get_stats_pa_representation(self) -> pd.DataFrame:
        """
        Returns each wearable's statistical measures 
        per hour (wearable.get_time_col()) 
        per day (wearable.get_experiment_day_col()) 

        :return: a dataframe with statistical measures by day and hour of the experiment
        """
        rows = []
        for wearable in self.wearables:
            act_hour = \
                wearable.data.groupby(
                    [wearable.get_experiment_day_col(), wearable.data[wearable.get_time_col()].dt.hour])[
                    wearable.get_activity_col()]

            rows.append(pd.DataFrame({"pid": wearable.get_pid(), "mean": act_hour.mean(),
                                      "median": act_hour.median(),
                                      "std": act_hour.std(),
                                      "min": act_hour.min(),
                                      "max": act_hour.max(),
                                      "skewness": act_hour.skew(),
                                      "kurtosis": act_hour.apply(pd.Series.kurt),
                                      "nunique": act_hour.nunique(),
                                      # Could give an indication of variability in the group
                                      }))
        return pd.concat(rows)

    def get_raw_pa(self, resolution):
        """
        Returns each wearable's raw physical activity either grouped by hour or day (resolution).
            per hour (wearable.get_time_col())
            per day (wearable.get_experiment_day_col())

        :param resolution:               Either "day" or "hour". The resolution expected for output.
        :return: a dataframe with raw physical activity
        """

        rows = []
        for wearable in self.wearables:
            if resolution == "hour":
                dfw = wearable.data.copy()
                dfw["minute"] = dfw[wearable.get_time_col()].dt.minute
                dfw["second"] = dfw[wearable.get_time_col()].dt.second
                activity = dfw.groupby([wearable.get_experiment_day_col(), dfw[wearable.get_time_col()].dt.hour, "minute", "second"])[
                    wearable.get_activity_col()]

                activity = activity.apply(lambda x: x.values.mean())
                activity = activity.reset_index().pivot(index=["ml_sequence", wearable.get_time_col()], columns=["minute", "second"])
                activity.columns = ['_'.join(map(lambda x: str(x).zfill(2), col)) for col in activity.columns.values]
                #activity = activity.apply(lambda x: np.vstack(x).reshape(-1), axis=1)
                #activity.name = "raw_pa"
                # activity = wearable.data.groupby([wearable.get_experiment_day_col(), wearable.data[wearable.get_time_col()].dt.hour])[wearable.get_activity_col()]

            elif resolution == "day":
                activity = wearable.data.groupby([wearable.get_experiment_day_col()])[wearable.get_activity_col()]

                activity = activity.apply(lambda x: x.values.ravel())
                activity.name = "raw_pa"

            activity = activity.reset_index()
            activity["pid"] = wearable.get_pid()
            rows.append(activity)

        return pd.concat(rows)
