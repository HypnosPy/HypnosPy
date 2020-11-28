from hypnospy import Wearable
from hypnospy import Experiment
from hypnospy import misc

import pandas as pd


class PhysicalActivity(object):

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
        :param sleep_col:                If a valid binary colunm, we ignore bouts that happened when the value of this col is True.
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
                tmp_df = bouts.groupby([wearable.get_experiment_day_col()])[pa_col].count().reset_index()
            elif resolution == "hour":
                gbouts = bouts.set_index("hyp_time_col")
                tmp_df = gbouts.groupby([wearable.get_experiment_day_col(), gbouts.index.hour])[pa_col].count().reset_index()
            else:
                raise ValueError("The parameter 'resolution' can only be `day` or `hour`.")

            tmp_df["pid"] = wearable.get_pid()
            tmp_df["bout_length"] = length_in_minutes
            returning_df.append(tmp_df)

        return pd.concat(returning_df).reset_index(drop=True)

    def get_binned_pa_representation(self):
        """
        Counts the number of occurance of physical activity names (PhysicalActivity.names) 
        within wearable.get_activity_col(). PhysicalActivity.name boundaries are set in 
        PhysicalActivity.cutoffs

        :return: dataframe with pa counts 
        per hour (wearable.get_time_col()) 
        per day (wearable.get_experiment_day_col()) 
        per wearable id (wearable.get_pid())
        """
        rows = []
        for wearable in self.wearables:
            act_hour = \
                wearable.data.groupby(
                    [wearable.get_experiment_day_col(), wearable.data[wearable.get_time_col()].dt.hour])[
                    wearable.get_activity_col()]

            pid = wearable.get_pid()
            # lpa = act_hour.apply(lambda x: (x <= self.mvpa_value).sum())
            lpa = act_hour.apply(lambda x: (x <= self.cutoffs[0]).sum())
            lpa.name = "LPA"
            # mvpa = act_hour.apply(lambda x: x.between(self.mvpa_value, self.vpa_value).sum())
            mvpa = act_hour.apply(lambda x: x.between(self.cutoffs[0], self.cutoffs[1]).sum())
            mvpa.name = "MVPA"
            # vpa = act_hour.apply(lambda x: (x >= self.vpa_value).sum())
            vpa = act_hour.apply(lambda x: (x >= self.cutoffs[1]).sum())
            vpa.name = "VPA"

            concatenated = pd.concat([lpa, mvpa, vpa], axis=1)
            concatenated["pid"] = pid
            rows.append(concatenated)

        return pd.concat(rows)

    def get_stats_pa_representation(self):
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

    def get_raw_pa(self):
        #  TODO: actually, one interesting way to implement it is having a concept of experiment_day that starts from 1
        #   rather than the current experiment_day that start from the day of the week

        # rows = []
        # for wearable in self.wearables:
        #     act_hour = \
        #         wearable.data.groupby(
        #             [wearable.get_experiment_day_col(), wearable.data[wearable.get_time_col()].dt.hour])[
        #             wearable.get_activity_col()]
        #     #act_hour.apply(lambda x: x.values.ravel())
        #     #act_hour["pid"] = wearable.get_pid()
        #     #rows.append(act_hour)
        #     return act_hour
        # #return pd.concat(rows)
        pass
