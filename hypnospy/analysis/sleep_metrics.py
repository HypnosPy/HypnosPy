from hypnospy import Wearable
from hypnospy import Experiment
from hypnospy import misc

from sklearn import metrics

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
    def calculate_awakening(df, wake_col, wake_delay=0, normalize_per_hour=False, epochs_in_hour=0):

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
    def calculate_sri(prev_block, current_block, wake_col, normalize_per_hour=False, epochs_in_hour=0):

        if prev_block.shape[0] != current_block.shape[0]:
            print("Unable to calculate SRI.")
            return -1.0

        same = (prev_block[wake_col].values == current_block[wake_col].values).sum()
        sri = -100 + (200 / prev_block.shape[0]) * same

        return sri

    def compare(self, ground_truth, wake_sleep_alg, sleep_period_col=None):

        results = {}

        for wearable in self.wearables:
            print("Sleep Metrics for:", wearable.get_pid())
            df = wearable.data
            result = {}

            for day, block in df.groupby(wearable.experiment_day_col):

                # filter where we should calculate the sleep metric
                if sleep_period_col is not None:
                    block = block[block[sleep_period_col] == True]

                gt, pred = block[ground_truth], block[wake_sleep_alg]

                result["accuracy"] = metrics.accuracy_score(gt, pred)
                result["precision"] = metrics.precision_score(gt, pred)
                result["recall"] = metrics.recall_score(gt, pred)
                result["f1_score"] = metrics.f1_score(gt, pred)
                result["roc_auc"] = metrics.roc_auc_score(gt, pred)
                result["cohens_kappa"] = metrics.cohen_kappa_score(gt, pred)

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

        result = {}
        for wearable in self.wearables:
            print("Sleep Metrics for:", wearable.get_pid())
            df = wearable.data
            result[wearable.get_pid()] = {}
            wake_delay_in_epochs = wearable.get_epochs_in_min() * wake_delay_in_minutes
            first_day = True
            prev_block = None

            for day, block in df.groupby(wearable.experiment_day_col):

                # filter where we should calculate the sleep metric
                if sleep_period_col is not None:
                    block = block[block[sleep_period_col] == True]

                if metric.lower() in ["sleepefficiency", "se"]:
                    result[wearable.get_pid()][day] = SleepMetrics.calculate_sleep_efficiency(block, wake_col,
                                                                                              wake_delay_in_epochs)

                elif metric.lower() in ["awakening", "awakenings"]:
                    result[wearable.get_pid()][day] = SleepMetrics.calculate_awakening(block, wake_col,
                                                                                       wake_delay_in_epochs,
                                                                                       normalize_per_hour=False)

                elif metric == "awakeningIndex":
                    result[wearable.get_pid()][day] = SleepMetrics.calculate_awakening(block, wake_col,
                                                                                       wake_delay_in_epochs,
                                                                                       normalize_per_hour=True,
                                                                                       epochs_in_hour=wearable.get_epochs_in_hour())

                elif metric == "arousal":
                    result[wearable.get_pid()][day] = SleepMetrics.calculate_arousals(block, wake_col,
                                                                                      wake_delay_in_epochs,
                                                                                      normalize_per_hour=False)

                elif metric == "arousalIndex":
                    result[wearable.get_pid()][day] = SleepMetrics.calculate_arousals(block, wake_col,
                                                                                      wake_delay_in_epochs,
                                                                                      normalize_per_hour=True,
                                                                                      epochs_in_hour=wearable.get_epochs_in_hour())

                elif metric == "totalTimeInBed":
                    result[wearable.get_pid()][day] = block.shape[0] / wearable.get_epochs_in_hour()

                elif metric == "totalSleepTime":
                    result[wearable.get_pid()][day] = ((block[wake_col] == 0).sum()) / wearable.get_epochs_in_hour()

                elif metric == "totalWakeTime":
                    result[wearable.get_pid()][day] = (block[wake_col].sum()) / wearable.get_epochs_in_hour()

                elif metric.lower() in ["sri"]:
                    if first_day:
                        first_day = False
                        prev_day_id = day
                        prev_block = block
                        continue

                    result[wearable.get_pid()]["%d-%d" % (prev_day_id,day)] = SleepMetrics.calculate_sri(prev_block, block, wake_col)
                    prev_day_id = day
                    prev_block = block

                else:
                    ValueError("Metric %s is unknown." % metric)

        return result
