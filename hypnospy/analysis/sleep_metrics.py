from hypnospy import Wearable
from hypnospy import Experiment
from hypnospy import misc

from sklearn import metrics
import pandas as pd


class SleepMetrics(object):

    def __init__(self, input: {Experiment, Wearable}) -> None:

        if type(input) is Wearable:
            self.wearables = [input]
        elif type(input) is Experiment:
            self.wearables = input.get_all_wearables()

    @staticmethod
    def calculate_sleep_efficiency(df_in, sleep_wake_col: str, ignore_awakenings_smaller_than_X_epochs: int = 0) -> float:
        """
        This method calculates the sleep efficiency from an input dataset.
        The sleep efficiency is calculated on the ``sleep_wake_col``, which is the result of any Sleep/Wake algorithm (see SleepWakeAnalysis).
        The parameter ``ignore_awakenings_smaller_than_X_epochs`` is used to avoid small signal fluctuations from sleep to wake.

        :param df_in: (partial) dataset to be analysed.
        :param sleep_wake_col: Dataframe column for a Sleep/Wake algorithm. Sleep = 1, Wake = 0.
        :param ignore_awakenings_smaller_than_X_epochs: Ignores changes from sleep to wake if they are smaller than X epochs.
        :return: sleep quality from 0 - 100 (the higher the better sleep quality)
        """

        if ignore_awakenings_smaller_than_X_epochs == 0:
            return 100. * (df_in[sleep_wake_col].sum() / df_in.shape[0]) if df_in.shape[0] > 0 else 0

        else:
            # Avoid modifying the original values in the wake col
            df = df_in[[sleep_wake_col]].copy()
            df["consecutive_state"], _ = misc.get_consecutive_series(df, sleep_wake_col)

            # If number of wakes (= 0) is smaller than X epochs, convert them to sleep (1):
            df.loc[(df[sleep_wake_col] == 0) & (
                    df["consecutive_state"] <= ignore_awakenings_smaller_than_X_epochs), sleep_wake_col] = 1
            sleep_quality = 100. * (df[sleep_wake_col].sum() / df.shape[0]) if df_in.shape[0] > 0 else 0
            return sleep_quality


    @staticmethod
    def calculate_awakening(df_in: pd.DataFrame, sleep_wake_col: str, ignore_awakenings_smaller_than_X_epochs: int = 0,
                            normalize_per_hour: bool = False, epochs_in_hour: int = 0) -> int:
        """
        This method calculates the number of awakenings (changes from sleep to wake stage).
        It uses the ``sleep_wake_col`` for that, which is the result of any Sleep/Wake algorithm (see SleepWakeAnalysis).
        The parameter ``ignore_awakenings_smaller_than_X_epochs`` is used to avoid small signal fluctuations from sleep to wake.


        :param df_in: (partial) dataset to be analysed.
        :param sleep_wake_col: Dataframe column for a Sleep/Wake algorithm. Sleep = 1, Wake = 0.
        :param ignore_awakenings_smaller_than_X_epochs: Ignores changes from sleep to wake if they are smaller than X epochs.
        :param normalize_per_hour: controls if the result should be normalized per hour of sleep or not
        :param epochs_in_hour: if ``normalize_per_hour`` is True, this parameter used in the normalization.
        :return: Number of awakenings in the df_in[sleep_wake_col] (normalized per hour if ``normalize_per_hour`` is True.
        """
        df = df_in.copy()

        df["consecutive_state"], df["gids"] = misc.get_consecutive_series(df, sleep_wake_col)
        # We ignore the first group of awakenings, as this method is only interested to count the number
        # of sequequencies after the subject slept for the first time.
        if df[(df["gids"] == 0) & (df[sleep_wake_col] == 0)].shape[0] > 0:
            df = df[(df["gids"] > 0)]

        grps = df[(df[sleep_wake_col] == 0) & (df["consecutive_state"] > ignore_awakenings_smaller_than_X_epochs)].groupby("gids")
        del df["consecutive_state"]
        del df["gids"]

        if normalize_per_hour:
            total_hours_slept = df.shape[0] / epochs_in_hour
            return len(grps) / total_hours_slept
        else:
            return len(grps)


    @staticmethod
    def calculate_sri(prev_block: pd.DataFrame, current_block: pd.DataFrame, sleep_wake_col: str) -> pd.DataFrame:
        """
        This method calculates the sleep regularity index of two dataframes, ``prev_block`` and ``current_block``,
        contrasting the ``sleep_wake_col`` values of these two dataframes.

        :param prev_block: first dataset to be analysed.
        :param current_block: second dataset to be analysed.
        :param sleep_wake_col: Dataframe column for a Sleep/Wake algorithm. Sleep = 1, Wake = 0.
        :return: the sleep regularity index (0-100). The higher the better regularity.
        """
        if prev_block.shape[0] != current_block.shape[0]:
            raise ValueError("Unable to calculate SRI.")

        same = (prev_block[sleep_wake_col].values == current_block[sleep_wake_col].values).sum()
        sri = (same / prev_block.shape[0]) * 100.

        return sri

    def get_sleep_quality(self, sleep_metric: str, wake_sleep_col: str, sleep_period_col:str = None,
                          outputname:str = None, ignore_awakenings_smaller_than_X_minutes: int = 0,
                          normalize_per_hour: bool = False) -> pd.DataFrame:
        """
        This method implements many different notions of sleep quality. Use ``sleep_metric`` to chose one of the many implemented here (see below).



        ``sleep_metric`` can be any of:
            - sleepEfficiency (0-100): the percentage of time slept in the dataframe
            - awakenings (> 0): counts the number of awakenings in the period. An awakening is a sequence of wake=1 periods larger than th_awakening (default = 10)
            - totalTimeInBed (in hours)
            - totalSleepTime (in hours)
            - totalWakeTime (in hours)
            - sri (Sleep Regularity Index, in percentage %)

        :param sleep_metric: sleep quality metric to be calculated.
        :param sleep_wake_col: Dataframe column for a Sleep/Wake algorithm. Sleep = 1, Wake = 0.
        :param sleep_period_col: Dataframe column for the actual sleep period (see SleepBoundaryDetector module)
        :param outputname: Name for the metric in the returned dataframe. Default: the metric used as ``sleep_metric``.
        :param ignore_awakenings_smaller_than_X_minutes: Ignores changes from sleep to wake if they are smaller than X epochs. Used in sleepEficiency and awakenings.
        :param normalize_per_hour: controls if the result should be normalized per hour of sleep or not. Used when the sleep_metric is awakenings.
        :return: A dataframe with 4 columns: <pid, exp_day_col, metric_name, parameters>.
                 Every row is the result of applying the sleep_metric on an experiment day for a given pid.
        """

        if outputname is None:
            outputname = sleep_metric

        results = []
        for wearable in self.wearables:
            df = wearable.data
            ignore_awakening_in_epochs = wearable.get_epochs_in_min() * ignore_awakenings_smaller_than_X_minutes
            first_day = True
            prev_block = None

            for day, block in df.groupby(wearable.get_experiment_day_col()):

                row = {"pid": wearable.get_pid(), wearable.get_experiment_day_col(): day}

                # filter where we should calculate the sleep metric
                if sleep_period_col is not None:
                    block = block[block[sleep_period_col] == True]

                if sleep_metric.lower() in ["sleepefficiency", "se", "sleep_efficiency"]:
                    row[outputname] = SleepMetrics.calculate_sleep_efficiency(block, wake_sleep_col,
                                                                           ignore_awakening_in_epochs)
                    row[outputname + "_parameters"] = {"ignore_awakening_in_epochs": ignore_awakening_in_epochs}

                elif sleep_metric.lower() in ["awakening", "awakenings", "arousal", "arousals"]:
                    row[outputname] = SleepMetrics.calculate_awakening(block, wake_sleep_col,
                                                                    ignore_awakening_in_epochs,
                                                                    normalize_per_hour=normalize_per_hour,
                                                                    epochs_in_hour=wearable.get_epochs_in_hour())
                    row[outputname + "_parameters"] = {"ignore_awakening_in_epochs": ignore_awakening_in_epochs,
                                         "normalize_per_hour": normalize_per_hour,
                                         "epochs_in_hour":wearable.get_epochs_in_hour()}

                elif sleep_metric == "totalTimeInBed":
                    row[outputname] = block.shape[0] / wearable.get_epochs_in_hour()
                    row[outputname + "_parameters"] = {}

                elif sleep_metric == "totalSleepTime":
                    row[outputname] = (block[wake_sleep_col].sum()) / wearable.get_epochs_in_hour()
                    row[outputname + "_parameters"] = {}

                elif sleep_metric == "totalWakeTime":
                    row[outputname] = ((block[wake_sleep_col] == 0).sum()) / wearable.get_epochs_in_hour()
                    row[outputname + "_parameters"] = {}

                elif sleep_metric.lower() in ["sri", "sleep_regularity_index", "sleepregularityindex"]:
                    if first_day:
                        first_day = False
                        prev_block = block
                        continue

                    try:
                        sri = SleepMetrics.calculate_sri(prev_block, block, wake_sleep_col)
                    except ValueError:
                        print("Unable to calculate SRI for day %d (PID = %s)." % (day, wearable.get_pid()))
                        sri = None
                    row[outputname] = sri
                    row[outputname + "_parameters"] = {}
                    prev_block = block

                else:
                    raise ValueError("Metric %s is unknown." % sleep_metric)

                results.append(row)

        return pd.DataFrame(results)

    def compare_sleep_metrics(self, ground_truth: str, sleep_wake_col: str, sleep_metrics: list,
                              sleep_period_col: str = None, comparison_method: str = "relative_difference") -> pd.DataFrame:
        """
        This method is used to compare a set of sleep_metrics based on a wake_sleep_col with a ground truth column (e.g., a column with PSG staging information).
        There are currently two different comparison methods (``comparison_method``) implemented:

            - "relative_difference" will compute the relative difference for each <pid, expday> for the ground_truth
               with the values for the same <pid, expday> entry of another `sleep_wake_col` method.
               This comparison is done with simply checking the delta difference between both values: 100 * (f(ground_truth) - f(value)) / f(value)

            - "pearson": simply uses the pearson correlation between the results of a sleep_metric for using a sleep_wake_col and the ground truth.
               This will result on a single value for each sleep_metric to be compared.

        ``sleep_metrics`` can be any of:
            - SleepEfficiency (0-100): the percentage of time slept in the dataframe
            - Awakenings (> 0): counts the number of awakenings in the period. An awakening is a sequence of wake=1 periods larger than th_awakening (default = 10)
            - TotalTimeInBed (in hours)
            - TotalSleepTime (in hours)
            - TotalWakeTime (in hours)
            - SRI (Sleep Regularity Index, in percentage %)

        `
        :param ground_truth: Ground Truth data to be compared with sleep_metric[sleep_wake_col] when sleep_perdiod_col == True
        :param sleep_wake_col: Dataframe column for a Sleep/Wake algorithm. Sleep = 1, Wake = 0.
        :param sleep_metrics: a list of sleep quality metrics to be compared.
        :param sleep_period_col: Dataframe column for the actual sleep period (see SleepBoundaryDetector module)
        :param comparison_method: how the comparison results should be reported. Options are "relative_difference" and "pearson"

        :return: a dataframe with the comparison results. Key differ according to the comparison method used.
        """

        results = []
        for sleep_metric in sleep_metrics:

            gt = self.get_sleep_quality(wake_sleep_col=ground_truth, sleep_metric=sleep_metric,
                                        sleep_period_col=sleep_period_col)
            other = self.get_sleep_quality(wake_sleep_col=sleep_wake_col, sleep_metric=sleep_metric,
                                           sleep_period_col=sleep_period_col)

            merged = pd.merge(gt[["pid", "hyp_exp_day", sleep_metric]], other[["pid", "hyp_exp_day", sleep_metric]],
                              on=["pid", "hyp_exp_day"], suffixes=["_gt", "_other"])

            if comparison_method == "relative_difference":
                merged["value"] = merged[[sleep_metric + "_gt", sleep_metric + "_other"]].apply(
                    lambda x: ((x[sleep_metric + "_gt"] - x[sleep_metric + "_other"]) /x[sleep_metric + "_other"]) * 100. if
                                    x[sleep_metric + "_other"] is not None and x[sleep_metric + "_other"] > 0 else 0, axis=1)
                merged["metric"] = "delta_" + sleep_metric
                merged["alg1"] = ground_truth
                merged["alg2"] = sleep_wake_col
                del merged[sleep_metric + "_gt"]
                del merged[sleep_metric + "_other"]

                results.append(merged)

            elif comparison_method == "pearson":
                value = merged[[sleep_metric + "_gt", sleep_metric + "_other"]].corr("pearson")[sleep_metric + "_gt"][
                    sleep_metric + "_other"]

                s = pd.Series({"value": value, "metric": "pearson_" + sleep_metric,
                               "alg1": ground_truth, "alg2": sleep_wake_col})
                results.append(s)

        if comparison_method == "relative_difference":
            concated = pd.concat(results)
            return concated

        else:
            return pd.concat(results, axis=1).T

    def evaluate_sleep_metric(self, ground_truth: str, sleep_wake_col: str, sleep_period_col: str = None) -> pd.DataFrame:
        """
        This method is used to compare the results obtained by a sleep_wake algorithm with a gronud truth (or another sleep_wake algorithm).
        The results are in terms of accuracy, precision, recall, f1_score, roc_auc and cohen's kappa.


        :param ground_truth: Ground Truth data to be compared with sleep_metric[sleep_wake_col] when sleep_perdiod_col == True
        :param sleep_wake_col: Dataframe column for a Sleep/Wake algorithm. Sleep = 1, Wake = 0.
        :param sleep_period_col: Dataframe column for the actual sleep period (see SleepBoundaryDetector module)

        :return: a dataframe with the comparison results. Key differ according to the comparison method used.
        """

        results = []

        # Usual evaluation metrics such as Accuracy, Precision, F1....
        for wearable in self.wearables:
            df = wearable.data

            for day, block in df.groupby(wearable.get_experiment_day_col()):

                # Filter where we should calculate the sleep metric
                if sleep_period_col is not None:
                    block = block[block[sleep_period_col] == True]

                if block.empty:
                    continue

                gt, pred = block[ground_truth], block[sleep_wake_col]

                result = {}
                result["accuracy"] = metrics.accuracy_score(gt, pred)
                result["precision"] = metrics.precision_score(gt, pred)
                result["recall"] = metrics.recall_score(gt, pred)
                result["f1_score"] = metrics.f1_score(gt, pred)
                result["roc_auc"] = metrics.roc_auc_score(gt, pred)
                result["cohens_kappa"] = metrics.cohen_kappa_score(gt, pred)
                result["pid"] = wearable.get_pid()
                result[wearable.get_experiment_day_col()] = day

                results.append(result)

        return pd.DataFrame(results)
