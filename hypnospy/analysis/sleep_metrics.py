from hypnospy import Wearable

import numpy as np
import pandas as pd
from hypnospy import misc


class SleepMetrics(object):

    def __init__(self, wearable: Wearable):
        self.wearable = wearable
        self.wake_col = None

    def set_wake_col(self, col):
        self.wake_col = col

    def get_wake_col(self, wake_col=None):
        if wake_col is None:
            if self.wake_col is None:
                ValueError("Need to setup ``wake_col`` before using this function.")
            else:
                return self.wake_col
        else:
            return wake_col

    def __get_sleep_efficiency(self, df, wake_col=None, wake_delay_min=0):
        wake_col = self.get_wake_col(wake_col)

        if wake_delay_min == 0:
            return 100 * (1. - df[wake_col].sum() / df.shape[0])

        else:
            if self.wearable.get_frequency_in_secs() == 30:
                wake_delay_epochs = 2 * wake_delay_min
            elif self.wearable.get_frequency_in_secs() == 60:
                wake_delay_epochs = wake_delay_min

            # Avoid modifying the original values in the wake col
            df["tmp_wake"] = df[wake_col].copy()
            df["consecutive_state"], _ = misc.get_consecutive_serie(df, wake_col)
            # 5 minutes = 10 entries counts
            # Change wake from 1 to 0 if group has less than 10 entries (= 5min)
            df.loc[(df["tmp_wake"] == 1) & (df["consecutive_state"] <= wake_delay_epochs), "tmp_wake"] = 0
            sleep_quality = 100 * (1. - df["tmp_wake"].sum() / df.shape[0])
            # delete aux cols
            del df["consecutive_state"]
            del df["tmp_wake"]
            return sleep_quality

    def __get_awakening(self, df, wake_col=None, wake_delay=0, normalize_per_hour=False):

        wake_col = self.get_wake_col(wake_col)

        df["consecutive_state"], df["gids"] = misc.get_consecutive_serie(df, wake_col)

        grps = df[(df[wake_col] == 1) & (df["consecutive_state"] > wake_delay)].groupby("gids")
        del df["consecutive_state"]
        del df["gids"]

        if normalize_per_hour:
            epochs_in_hour = self.wearable.get_epochs_in_hour()
            total_hours_slept = df.shape[0] / epochs_in_hour
            return len(grps) / total_hours_slept
        else:
            return len(grps)

    def __get_arousals(self, df, wake_col=None, normalize_per_hour=False):

        wake_col = self.get_wake_col(wake_col)

        arousals = ((df[wake_col] == 1) & (df[wake_col] != df[wake_col].shift(1).fillna(0))).sum()

        if normalize_per_hour:
            epochs_in_hour = self.wearable.get_epochs_in_hour()
            total_hours_slept = df.shape[0] / epochs_in_hour
            return arousals / total_hours_slept
        else:
            return arousals

    def get_sleep_quality(self, wake_col=None, sleep_period_col=None,
                          strategy="sleepEfficiencyAll", wake_delay=10):
        """
            This function implements different notions of sleep quality.
            For far strategy can be:
            - sleepEfficiencyAll (0-100): the percentage of time slept (wake=0) in the dataframe
            - sleepEfficiency5min (0-100): similar to above but only considers wake=1 if the awake period is longer than th_awakening (default = 10)
            - awakening (> 0): counts the number of awakenings in the period. An awakening is a sequence of wake=1 periods larger than th_awakening (default = 10)
            - awakeningIndex (> 0)
            - arousal (> 0):
            - arousalIndex (>0):
            - totalTimeInBed (in hours)
            - totalSleepTime (in hours)
            - totalWakeTime (in hours)
            - SRI (Sleep Regularity Index, in percentage %)
        """

        wake_col = self.get_wake_col()

        # if sleep_period_col: # TODO: need to filter.
        df = self.wearable.data

        if strategy == "sleepEfficiencyAll":
            return self.__get_sleep_efficiency(df, wake_col, 0)

        elif strategy == "sleepEfficiency5min":
            return self.__get_sleep_efficiency(df, wake_col, wake_delay)

        elif strategy == "awakening":
            return self.__get_awakening(df, wake_col, wake_delay, normalize_per_hour=False)

        elif strategy == "awakeningIndex":
            return self.__get_awakening(df, wake_col, wake_delay, normalize_per_hour=True)

        elif strategy == "arousal":
            return self.__get_arousals(df, wake_col, wake_delay, normalize_per_hour=False)

        elif strategy == "arousalIndex":
            return self.__get_arousals(df, wake_col, wake_delay, normalize_per_hour=True)

        elif strategy == "totalTimeInBed":
            return df.shape[0] / 60. #TODO: should we divide it by 60?

        elif strategy == "totalSleepTime":
            return ((df[wake_col] == 0).sum()) / 60.

        elif strategy == "totalWakeTime":
            return (df[wake_col].sum()) / 60.

        elif strategy == "SRI":
            # TODO: missing
            # SRI needs some tuning
            # df = data.between_time(bed_time,wake_time, include_start = True,
            # include_end = True) Change this to sleep search window
            sri_delta = np.zeros(len(df[df.index[0]:df.shift(periods=-1, freq='D').index[-1]]))
            for i in range(len(df[df.index[0]:df.shift(periods=-1, freq='D').index[-1]])):
                if df[sleep_col][df.index[i]] == df[sleep_col].shift(periods=-1, freq='D')[df.index[i]]:
                    sri_delta[i] = 1
                else:
                    sri_delta[i] = 0
            sri_df = pd.DataFrame(sri_delta)
            sri = -100 + (200 / (len(df[df.index[0]:df.shift(periods=-1, freq='D').index[-1]]))) * sri_df.sum()
            return float(sri)

        else:
            ValueError("Strategy %s is unknown." % (strategy))


    # %%
    def get_sleep_metrics(df_in,
                          min_sleep_block,
                          sleep_block_col,
                          sleep_metrics,
                          wake_sleep_col="Wake_Sleep",
                          timecol="ts",
                          what_is_sleep_value=False
                          # The default is sleep = 0, set this to True if in your dataset sleep = 1
                          ):

        df = df_in[df_in[sleep_block_col] >= min_sleep_block].copy()

        if what_is_sleep_value:
            df[wake_sleep_col] = ~df[wake_sleep_col]

        series = []
        for sleep_metric in sleep_metrics:
            r = df[[sleep_block_col,
                    wake_sleep_col]].groupby(sleep_block_col).apply(lambda grp: sleep_quality_from_wake_col(grp,
                                                                                                            wake_sleep_col,
                                                                                                            strategy=sleep_metric))
            r.name = sleep_metric
            series.append(r)

        return pd.concat(series, axis=1).reset_index()
