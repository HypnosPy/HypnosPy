from hypnospy import Wearable
from hypnospy import Experiment
from hypnospy import misc

import pandas as pd


class PhysicalActivity(object):

    # TODO: activity is currently limited to one axis.

    def __init__(self, input: {Experiment, Wearable}, lpa=None, mvpa=None, vpa=None):

        if type(input) is Wearable:
            self.wearables = [input]
        elif type(input) is Experiment:
            self.wearables = input.get_all_wearables()

        self.sed_col = "hyp_sed"
        self.lpa_col = "hyp_lpa"
        self.mvpa_col = "hyp_mvpa"
        self.vpa_col = "hyp_vpa"

        self.lpa_value = self.mvpa_value = self.vpa_value = None
        self.set_pa_thresholds(lpa, mvpa, vpa)

    def set_pa_thresholds(self, lpa=None, mvpa=None, vpa=None):
        if lpa is not None:
            self.lpa_value = lpa
        if mvpa is not None:
            self.mvpa_value = mvpa
        if vpa is not None:
            self.vpa_value = vpa

    def generate_pa_columns(self, based_on="activity"):
        if self.lpa_value is None or self.mvpa_value is None or self.vpa_value is None:
            raise AttributeError("Please use `set_pa_thresholds` before using this method.")

        for wearable in self.wearables:
            if based_on.lower() == "activity":
                col = wearable.get_activity_col()
            elif based_on.lower() == "mets":
                col = wearable.get_mets_col()
            else:
                col = based_on

            # TODO: If we have a value that is VPA, is it also MVPA and LPA?
            wearable.data[self.sed_col] = wearable.data[col] <= self.lpa_value
            wearable.data[self.lpa_col] = wearable.data[col] > self.lpa_value
            wearable.data[self.mvpa_col] = wearable.data[col] > self.mvpa_value
            wearable.data[self.vpa_col] = wearable.data[col] > self.vpa_value
            wearable.pa_intensity_cols = [self.sed_col, self.lpa_col, self.mvpa_col, self.vpa_col]

    def __pa_cols_okay(self):
        for wearable in self.wearables:
            keys = wearable.data.keys()
            if self.sed_col not in keys or self.lpa_col not in keys or \
                    self.mvpa_col not in keys or self.vpa_col not in keys:
                return False
        return True

    def mask_sleep_period(self, mask):
        # TODO: missing implementation and finding a better name for this method.
        df["X"] = df["activity"].where(~df["sleep_period"].astype(np.bool), 0)

    def __get_bout(self, dd, epochs_per_minute, pacol, mins=10, decomposite_bouts=True):

        dd["pa_len"], dd["pa_grp"] = misc.get_consecutive_serie(dd, pacol)
        bouts = dd[(dd[pacol] == True) & (dd["pa_len"] >= mins * epochs_per_minute)]

        if decomposite_bouts:
            # Should we break a 20 min bout in 2 of 10 mins?
            bouts["num_sub_bouts"] = bouts["pa_len"] // (mins * epochs_per_minute)
            return bouts.groupby("pa_grp").first()["num_sub_bouts"].sum()
        else:
            return len(bouts["pa_grp"].unique())

    def get_vpas(self, length_in_minutes, decomposite_bouts=False):
        if self.vpa_value is None:
            raise ValueError("Please set VPA first by running ``set_pa_thresholds``.")
        if not self.__pa_cols_okay():
            raise AttributeError("Please use ``generate_pa_columns`` before using this function.")

        returning_dict = {}
        for wearable in self.wearables:
            pid = wearable.get_pid()
            epochs_per_minute = wearable.get_epochs_in_min()
            returning_dict[pid] = wearable.data.groupby(wearable.experiment_day_col).apply(
                lambda x: self.__get_bout(x, epochs_per_minute, pacol=self.vpa_col, mins=length_in_minutes,
                                          decomposite_bouts=decomposite_bouts)
            )
        return returning_dict

    def get_mvpas(self, length_in_minutes, decomposite_bouts=False):
        if self.mvpa_value is None:
            raise ValueError("Please set MVPA first by running ``set_pa_thresholds``.")
        if not self.__pa_cols_okay():
            raise AttributeError("Please use ``generate_pa_columns`` before using this function.")

        returning_dict = {}
        for wearable in self.wearables:
            pid = wearable.get_pid()
            epochs_per_minute = wearable.get_epochs_in_min()
            returning_dict[pid] = wearable.data.groupby(wearable.experiment_day_col).apply(
                lambda x: self.__get_bout(x, epochs_per_minute, pacol=self.mvpa_col, mins=length_in_minutes,
                                          decomposite_bouts=decomposite_bouts)
            )
        return returning_dict

    def get_lpas(self, length_in_minutes, decomposite_bouts=False):
        if self.mvpa_value is None:
            raise ValueError("Please set LPA first by running ``set_pa_thresholds``. ")
        if not self.__pa_cols_okay():
            raise AttributeError("Please use ``generate_pa_columns`` before using this function.")

        returning_dict = {}
        for wearable in self.wearables:
            pid = wearable.get_pid()
            epochs_per_minute = wearable.get_epochs_in_min()
            returning_dict[pid] = wearable.data.groupby(wearable.experiment_day_col).apply(
                lambda x: self.__get_bout(x, epochs_per_minute, pacol=self.lpa_col, mins=length_in_minutes,
                                          decomposite_bouts=decomposite_bouts)
            )
        return returning_dict

    def get_seds(self, length_in_minutes, decomposite_bouts=False):
        if self.lpa_value is None:
            raise ValueError("Please set SED first by running ``set_pa_thresholds``. "
                             "SED will be any value equal or small than LPA. ")

        returning_dict = {}
        for wearable in self.wearables:
            pid = wearable.get_pid()
            epochs_per_minute = wearable.get_epochs_in_min()
            returning_dict[pid] = wearable.data.groupby(wearable.experiment_day_col).apply(
                lambda x: self.__get_bout(x, epochs_per_minute, pacol=self.sed_col, mins=length_in_minutes,
                                          decomposite_bouts=decomposite_bouts)
            )
        return returning_dict

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
            lpa = act_hour.apply(lambda x: (x <= self.mvpa_value).sum())
            lpa.name = "LPA"
            mvpa = act_hour.apply(lambda x: x.between(self.mvpa_value, self.vpa_value).sum())
            mvpa.name = "MVPA"
            vpa = act_hour.apply(lambda x: (x >= self.vpa_value).sum())
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
