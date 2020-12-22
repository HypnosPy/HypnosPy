import pandas as pd
import numpy as np
from hypnospy import misc, Wearable, Experiment


class Demographics(object):

    def __init__(self, file_path: str, pid_col: str, clock_features: list = None, save_only_cols: list = None,
                 na_value: object = np.nan, dtype: object = np.float):

        self.data = pd.read_csv(file_path)

        if pid_col not in self.data.keys():
            raise KeyError("Pid col %s not found in the demographics. " % pid_col)

        self.data["pid"] = self.data[pid_col].astype(str)
        self.data.set_index("pid", inplace=True)

        if save_only_cols:
            try:
                self.data = self.data[save_only_cols]
            except KeyError:
                for k in save_only_cols:
                    if k not in self.data:
                        raise KeyError("\n%s\nError: Could not find key %s.\n%s\n" % ('*' * 80, k, '*' * 80))

        # Fix problems with demographic data
        self._convert_clock_features(clock_features)

        if dtype:
            for col in self.data.keys():
                self.data[col] = self.data[col].astype(dtype)

        # Fill na if needed
        self.data.fillna(na_value, inplace=True)

    def _convert_clock_features(self, clock_features: list) -> None:
        if clock_features:
            for feature in clock_features:
                if feature in self.data:
                    self.data[feature] = self.data[feature].apply(lambda x: misc.convert_clock_to_sec_since_midnight(x))

    def add_to_wearable(self, wearable: Wearable) -> None:
        pid = wearable.get_pid()
        if pid not in self.data.index:
            raise KeyError("PID %s not found in the demographic data." % pid)

        wearable.demographics = self.data.loc[pid]

    def add_to_experiment(self, experiment: Experiment) -> None:

        for wearable in experiment.get_all_wearables():
            self.add_to_wearable(wearable)

    def get_valid_data_for_experiment(self, experiment: Experiment) -> pd.DataFrame:
        pids = [wearable.get_pid() for wearable in experiment.get_all_wearables()]
        return self.data[self.data.index.isin(pids)]
