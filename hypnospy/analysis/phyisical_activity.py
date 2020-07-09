from hypnospy import Wearable
from hypnospy import Experiment
from hypnospy import misc

import pandas as pd


class PhysicalActivity(object):

    # TODO: activity is currently limited to one axis.

    def __init__(self, input: {Experiment, Wearable}, mvpa=None, vpa=None):

        if type(input) is Wearable:
            self.wearables = [input]
        elif type(input) is Experiment:
            self.wearables = input.get_all_wearables()

        # self.activity_col = wearable.activitycols[0]
        # self.experiment_day_col = self.wearable.experiment_day_col
        # self.time_col = self.wearable.time_col

        self.mvpa_value = self.vpa_value = None
        self.set_pa_thresholds(mvpa, vpa)

    def set_pa_thresholds(self, mvpa=None, vpa=None):
        if mvpa is not None:
            self.mvpa_value = mvpa

        if vpa is not None:
            self.vpa_value = vpa

    def mask_sleep_period(self, mask):
        # TODO:
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

    def get_mvpas(self, length_in_minutes, decomposite_bouts=False):
        if self.mvpa_value is None:
            raise ValueError("Please set MVPA first by running ``set_pa_thresholds``.")

        returning_dict = {}
        for wearable in self.wearables:
            wearable.data["hyp_mvpa"] = wearable.data[wearable.get_activity_col()] > self.mvpa_value
            pid = wearable.get_pid()
            epochs_per_minute = wearable.get_epochs_in_min()
            returning_dict[pid] = wearable.data.groupby(wearable.experiment_day_col).apply(
                lambda x: self.__get_bout(x, epochs_per_minute, pacol="hyp_mvpa", mins=length_in_minutes,
                                          decomposite_bouts=decomposite_bouts)
            )
        return returning_dict

    def get_vpas(self, length_in_minutes, decomposite_bouts=False):
        if self.vpa_value is None:
            raise ValueError("Please set VPA first by running ``set_pa_thresholds``.")

        returning_dict = {}
        for wearable in self.wearables:
            wearable.data["hyp_vpa"] = wearable.data[wearable.get_activity_col()] > self.vpa_value
            pid = wearable.get_pid()
            epochs_per_minute = wearable.get_epochs_in_min()
            returning_dict[pid] = wearable.data.groupby(wearable.experiment_day_col).apply(
                lambda x: self.__get_bout(x, epochs_per_minute, pacol="hyp_vpa", mins=length_in_minutes,
                                          decomposite_bouts=decomposite_bouts)
            )
        return returning_dict

    def get_lpas(self, length_in_minutes, decomposite_bouts=False):
        if self.mvpa_value is None:
            raise ValueError("Please set MVPA first by running ``set_pa_thresholds``. "
                       "LPA will be any value equal or small than MVPA. ")

        returning_dict = {}
        for wearable in self.wearables:
            pid = wearable.get_pid()
            epochs_per_minute = wearable.get_epochs_in_min()
            wearable.data["hyp_lpa"] = wearable.data[wearable.get_activity_col()] <= self.mvpa_value

            returning_dict[pid] = wearable.data.groupby(wearable.experiment_day_col).apply(
                lambda x: self.__get_bout(x, epochs_per_minute, pacol="hyp_lpa", mins=length_in_minutes,
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
                wearable.data.groupby([wearable.get_experiment_day_col(), wearable.data[wearable.get_time_col()].dt.hour])[
                    wearable.get_activity_col()]

            pid = wearable.get_pid()
            lpa = act_hour.apply(lambda x: (x <= self.mvpa_value).sum())
            lpa.name = "LPA"
            mvpa = act_hour.apply(lambda x: x.between(self.mvpa_value, self.vpa_value).sum())
            mvpa.name = "MVPA"
            vpa = act_hour.apply(lambda x: (x >= self.vpa_value).sum())
            vpa.name = "VPA"
            concatted = pd.concat([lpa, mvpa, vpa], axis=1)
            concatted["pid"] = pid
            rows.append(concatted)

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

        #rows = []
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
