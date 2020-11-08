from hypnospy import Wearable
from hypnospy import Experiment
from hypnospy import misc

from sklearn import metrics
import pandas as pd


class SleepMetrics(object):

    def __init__(self, input: {Experiment, Wearable}):

        if type(input) is Wearable:
            self.wearables = [input]
        elif type(input) is Experiment:
            self.wearables = input.get_all_wearables()

    @staticmethod
    def calculate_sleep_efficiency(df_in, wake_col, wake_delay_epochs=0):
        if wake_delay_epochs == 0:
            return 100 * (1. - df_in[wake_col].sum() / df_in.shape[0])

        else:
            # Avoid modifying the original values in the wake col
            df = df_in.copy()
            df["consecutive_state"], _ = misc.get_consecutive_serie(df, wake_col)
            # Change wake from 1 to 0 if group has less than ``wake_delay_epochs`` entries
            df.loc[(df[wake_col] == 1) & (df["consecutive_state"] <= wake_delay_epochs), wake_col] = 0
            sleep_quality = 100 * (1. - df[wake_col].sum() / df.shape[0])
            # delete aux cols
            return sleep_quality

    @staticmethod
    def calculate_awakening(df_in, wake_col, wake_delay=0, normalize_per_hour=False, epochs_in_hour=0):
        df = df_in.copy()

        df["consecutive_state"], df["gids"] = misc.get_consecutive_serie(df, wake_col)

        grps = df[(df[wake_col] == 1) & (df["consecutive_state"] > wake_delay)].groupby("gids")
        del df["consecutive_state"]
        del df["gids"]

        if normalize_per_hour:
            total_hours_slept = df.shape[0] / epochs_in_hour
            return len(grps) / total_hours_slept
        else:
            return len(grps)

    @staticmethod
    def calculate_arousals(df, wake_col=None, normalize_per_hour=False, epochs_in_hour=0):

        arousals = ((df[wake_col] == 1) & (df[wake_col] != df[wake_col].shift(1).fillna(0))).sum()

        if normalize_per_hour:
            total_hours_slept = df.shape[0] / epochs_in_hour
            return arousals / total_hours_slept
        else:
            return arousals

    @staticmethod
    def calculate_sri(prev_block, current_block, wake_col):

        if prev_block.shape[0] != current_block.shape[0]:
            raise ValueError("Unable to calculate SRI.")

        same = (prev_block[wake_col].values == current_block[wake_col].values).sum()
        sri = (same / prev_block.shape[0]) * 100.

        return sri

    def compare_sleep_metrics(self, ground_truth, wake_sleep_alg, sleep_metrics=["sleepEfficiency", "awakening", "SRI",
                                                                                 "arousal", "totalTimeInBed",
                                                                                 "totalSleepTime", "totalWakeTime"],
                              how="relative_difference"):

        results = []
        for sleep_metric in sleep_metrics:

            gt = self.get_sleep_quality(wake_col=ground_truth, metric=sleep_metric)
            other = self.get_sleep_quality(wake_col=wake_sleep_alg, metric=sleep_metric)

            gtdf = pd.DataFrame(gt)
            otherdf = pd.DataFrame(other)
            merged = pd.merge(gtdf, otherdf, on=["pid", "expday", "metric"], suffixes=["_gt", "_other"])

            if how == "relative_difference":
                merged["value"] = merged[["value_gt", "value_other"]].apply(
                    lambda x: ((x["value_gt"] - x["value_other"]) / x["value_other"]) * 100. if x[
                                                                                                    "value_other"] is not None and
                                                                                                x[
                                                                                                    "value_other"] > 0 else 0,
                    axis=1)
                merged["metric"] = "delta_" + merged["metric"]
                merged["alg1"] = ground_truth
                merged["alg2"] = wake_sleep_alg
                del merged["value_gt"]
                del merged["value_other"]

                results.append(merged)

            elif how == "pearson":
                value = merged[["value_gt", "value_other"]].corr("pearson")["value_gt"]["value_other"]

                s = pd.Series({"value": value, "metric": "pearson_" + sleep_metric,
                               "alg1": ground_truth, "alg2": wake_sleep_alg})
                results.append(s)

        if how == "relative_difference":
            concated = pd.concat(results)
            return concated.to_dict('records')

        else:
            return pd.concat(results, axis=1).T.to_dict('records')

    def compare(self, ground_truth, wake_sleep_alg, sleep_period_col=None):

        results = {}

        # Usual evaluation metrics such as Accuracy, Precision, F1....
        for wearable in self.wearables:
            df = wearable.data
            result = {"accuracy": [], "precision": [], "recall": [], "f1_score": [], "roc_auc": [], "cohens_kappa": []}

            for day, block in df.groupby(wearable.experiment_day_col):

                # Filter where we should calculate the sleep metric
                if sleep_period_col is not None:
                    block = block[block[sleep_period_col] == True]

                gt, pred = block[ground_truth], block[wake_sleep_alg]

                result["accuracy"].append(metrics.accuracy_score(gt, pred))
                result["precision"].append(metrics.precision_score(gt, pred))
                result["recall"].append(metrics.recall_score(gt, pred))
                result["f1_score"].append(metrics.f1_score(gt, pred))
                result["roc_auc"].append(metrics.roc_auc_score(gt, pred))
                result["cohens_kappa"].append(metrics.cohen_kappa_score(gt, pred))

            results[wearable.get_pid()] = result

        return results

    def get_sleep_quality(self, wake_col, sleep_period_col=None, metric="sleepEfficiency", wake_delay_in_minutes=10):
        """
            This function implements different notions of sleep quality.
            For far strategy can be:
            - sleepEfficiency (0-100): the percentage of time slept (wake=0) in the dataframe
            - awakening (> 0): counts the number of awakenings in the period. An awakening is a sequence of wake=1 periods larger than th_awakening (default = 10)
            - awakeningIndex (> 0)
            - arousal (> 0):
            - arousalIndex (>0):
            - totalTimeInBed (in hours)
            - totalSleepTime (in hours)
            - totalWakeTime (in hours)
            - SRI (Sleep Regularity Index, in percentage %)
        """

        results = []
        for wearable in self.wearables:
            df = wearable.data
            wake_delay_in_epochs = wearable.get_epochs_in_min() * wake_delay_in_minutes
            first_day = True
            prev_block = None

            for day, block in df.groupby(wearable.experiment_day_col):

                row = {"pid": wearable.get_pid(), "expday": day}

                # filter where we should calculate the sleep metric
                if sleep_period_col is not None:
                    block = block[block[sleep_period_col] == True]

                if metric.lower() in ["sleepefficiency", "se"]:
                    row["metric"] = "sleepefficiency"
                    row["value"] = SleepMetrics.calculate_sleep_efficiency(block, wake_col,
                                                                           wake_delay_in_epochs)

                elif metric.lower() in ["awakening", "awakenings"]:
                    row["metric"] = "awakening"
                    row["value"] = SleepMetrics.calculate_awakening(block, wake_col,
                                                                    wake_delay_in_epochs,
                                                                    normalize_per_hour=False)

                elif metric == "awakeningIndex":
                    row["metric"] = "awakeningIndex"
                    row["value"] = SleepMetrics.calculate_awakening(block, wake_col, wake_delay_in_epochs,
                                                                    normalize_per_hour=True,
                                                                    epochs_in_hour=wearable.get_epochs_in_hour())

                elif metric == "arousal":
                    row["metric"] = "arousal"
                    row["value"] = SleepMetrics.calculate_arousals(block, wake_col, normalize_per_hour=False)

                elif metric == "arousalIndex":
                    row["metric"] = "arousalIndex"
                    row["value"] = SleepMetrics.calculate_arousals(block, wake_col, normalize_per_hour=True,
                                                                   epochs_in_hour=wearable.get_epochs_in_hour())
                elif metric == "totalTimeInBed":
                    row["metric"] = "totalTimeInBed"
                    row["value"] = block.shape[0] / wearable.get_epochs_in_hour()

                elif metric == "totalSleepTime":
                    row["metric"] = "totalSleepTime"
                    row["value"] = ((block[wake_col] == 0).sum()) / wearable.get_epochs_in_hour()

                elif metric == "totalWakeTime":
                    row["metric"] = "totalWakeTime"
                    row["value"] = (block[wake_col].sum()) / wearable.get_epochs_in_hour()

                elif metric.lower() in ["sri", "sleep_regularity_index", "sleepregularityindex"]:
                    if first_day:
                        first_day = False
                        prev_block = block
                        continue

                    row["metric"] = "sri"
                    try:
                        sri = SleepMetrics.calculate_sri(prev_block, block, wake_col)
                    except ValueError:
                        print("Unable to calculate SRI for day %d (PID = %s)." % (day, wearable.get_pid()))
                        sri = None
                    row["value"] = sri
                    prev_block = block

                else:
                    raise ValueError("Metric %s is unknown." % metric)

                results.append(row)

        return pd.DataFrame(results)

    def __evaluate_sleep_boundaries_pair(self, ground_truth, other):

        df_acc = []
        expid = 0

        for w in self.wearables:

            if w.data.empty:
                print("Data for PID %s is empty!" % w.get_pid())
                continue

            if ground_truth not in w.data:
                print("Column %s not in dataset for PID %s." % (ground_truth, w.get_pid()))
                continue

            if other not in w.data:
                print("Column %s not in dataset for PID %s." % (other, w.get_pid()))
                continue

            sleep = {}
            sleep[ground_truth] = w.data[ground_truth].astype(int)
            sleep[other] = w.data[other].astype(int)

            if sleep[ground_truth].shape[0] == 0:
                continue

            mse = metrics.mean_squared_error(sleep[ground_truth], sleep[other])
            cohen = metrics.cohen_kappa_score(sleep[ground_truth], sleep[other])

            tst_gt = w.get_total_sleep_time_per_day(sleep_col=ground_truth)
            tst_gt.rename(columns={ground_truth: "tst_" + ground_truth}, inplace=True)
            tst_other = w.get_total_sleep_time_per_day(sleep_col=other)
            tst_other.rename(columns={other: "tst_" + other}, inplace=True)

            onset_gt = w.get_onset_sleep_time_per_day(sleep_col=ground_truth)
            onset_gt.name = "onset_" + ground_truth
            onset_other = w.get_onset_sleep_time_per_day(sleep_col=other)
            onset_other.name = "onset_" + other

            offset_gt = w.get_offset_sleep_time_per_day(sleep_col=ground_truth)
            offset_gt.name = "offset_" + ground_truth
            offset_other = w.get_offset_sleep_time_per_day(sleep_col=other)
            offset_other.name = "offset_" + other

            df_res = pd.concat((onset_gt, onset_other, offset_gt, offset_other, tst_gt, tst_other), axis=1)
            df_res["pid"] = w.get_pid()
            df_res["mse_" + ground_truth + "&" + other] = mse
            df_res["cohens_" + ground_truth + "&" + other] = cohen
            df_res = df_res.reset_index()
            expid += 1

            df_acc.append(df_res)

        df_acc = pd.concat(df_acc)

        # Drop invalid rows:
        for col in ["onset_", "offset_", "tst_"]:
            df_acc = df_acc[~df_acc[col + ground_truth].isnull()]

        return df_acc

    def evaluate_sleep_boundaries(self, ground_truth, others):
        dfs = []
        for other in others:
            dfs.append(self.__evaluate_sleep_boundaries_pair(ground_truth, other))

        if len(dfs) == 0:
            return None

        result = dfs[0]
        for other in dfs[1:]:
            common_keys = list(set(result.keys()).intersection(other.keys()))
            result = result.merge(other, on=common_keys)

        return result
