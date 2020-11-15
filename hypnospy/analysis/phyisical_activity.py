from hypnospy import Wearable
from hypnospy import Experiment
from hypnospy import misc

import pandas as pd


class PhysicalActivity(object):

    def __init__(self, input: {Experiment, Wearable}, cutoffs=None, names=None):
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

    def get_bouts(self, pa_col, length_in_minutes, decomposite_bouts=False):
        """

        :param pa_col:              The name of the physical activity column in the dataframe.
        :param length_in_minutes:   The minimal length of the activity in minutes
        :param decomposite_bouts:   If True, we are going to count the number of subsequencies of ``length_in_minutes``.
                                    For example, if ``length_in_minutes`` is 10 and we identified a continuous activity of 30 minutes,
                                    if ``decomposite_bouts`` is True, we will count 3 bouts for this activity, otherwise only 1.
        :return:                    A dictionary with <pid> keys and Series <day, bouts> as values.
        """

        if pa_col not in self.names:
            raise ValueError("Unknown physical activity column %s. Please use ``set_cutoffs``.")

        returning_dict = {}
        for wearable in self.wearables:
            pid = wearable.get_pid()
            epochs_per_minute = wearable.get_epochs_in_min()
            returning_dict[pid] = wearable.data.groupby(wearable.experiment_day_col).apply(
                lambda x: self._get_bout(x, epochs_per_minute, pacol=pa_col, mins=length_in_minutes,
                                         decomposite_bouts=decomposite_bouts)
            )
        return returning_dict

    @staticmethod
    def _get_bout(dd, epochs_per_minute, pacol, mins=10, decomposite_bouts=True):

        dd["pa_len"], dd["pa_grp"] = misc.get_consecutive_serie(dd, pacol)
        bouts = dd[(dd[pacol] == True) & (dd["pa_len"] >= mins * epochs_per_minute)]

        if decomposite_bouts:
            # Should we break a 20 min bout in 2 of 10 mins?
            bouts["num_sub_bouts"] = bouts["pa_len"] // (mins * epochs_per_minute)
            return bouts.groupby("pa_grp").first()["num_sub_bouts"].sum()
        else:
            return len(bouts["pa_grp"].unique())

    def mask_sleep_period(self, mask):
        # TODO: missing implementation and finding a better name for this method.
        df["X"] = df["activity"].where(~df["sleep_period"].astype(np.bool), 0)

    def get_binned_pa_representation(self):
        """

        :return:
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
