from hypnospy import Wearable
from hypnospy import Experiment
import numpy as np
import warnings
from enum import IntFlag


class InvCode(IntFlag):
    # unfortunately the Flag type does not work properly with pandas 1.14
    FLAG_OKAY = 0
    FLAG_EPOCH_PA = 1
    FLAG_EPOCH_NON_WEARING = 2
    FLAG_EPOCH_NULL_VALUE = 4
    FLAG_DAY_SHORT_SLEEP = 8
    FLAG_DAY_LONG_SLEEP = 16
    FLAG_DAY_WITHOUT_DIARY = 32
    FLAG_DAY_NON_WEARING = 64
    FLAG_DAY_NOT_ENOUGH_VALID_EPOCHS = 128
    FLAG_DAY_NOT_ENOUGH_CONSECUTIVE_DAYS = 256

    @staticmethod
    def check_flag(int_value):
        return InvCode.FLAG_OKAY | int_value


class Validator(object):

    def __init__(self, input: {Wearable, Experiment}):
        """
        Two flagging levels: epoch and day.

        :param input:
        """
        self.wearables = {}

        if input is None:
            raise ValueError("Invalid value for input.")
        elif type(input) is Wearable:
            self.wearables[input.get_pid()] = input
        elif type(input) is Experiment:
            self.wearables = input.wearables

        self.invalid_col = "hyp_invalid"
        for wearable in self.wearables.values():
            wearable.data[self.invalid_col] = InvCode.FLAG_OKAY

    def flag_epoch_physical_activity_less_than(self, min_activity_threshold: int = 0):
        """
        Marks as invalid (InvCode.FLAG_PA) if physical activity is below ``min_activity_threshold``.

        :param min_activity_threshold: Integer threshold. Default: 0
        :return: None
        """
        for wearable in self.wearables.values():
            # Mark activity smaller than minimal
            wearable.data.loc[wearable.data[
                                  wearable.get_activity_col()] < min_activity_threshold, self.invalid_col] |= InvCode.FLAG_EPOCH_PA

    def flag_epoch_null_cols(self, col_list: list):

        for wearable in self.wearables.values():
            for col in col_list:
                if col not in wearable.data.keys():
                    raise KeyError("Col %s is not available for PID %s" % (col, wearable.get_pid()))

                wearable.data.loc[wearable.data[col].isnull(), self.invalid_col] |= InvCode.FLAG_EPOCH_NULL_VALUE

    def flag_epoch_nonwearing(self, wearing_col: str):
        """

        :param wearing_col:
        :return:
        """
        for wearable in self.wearables.values():
            if wearing_col not in wearable.data.keys():
                raise KeyError(
                    "Column %s not found for wearable (pid=%s). Did you forget to run ``NonWearingDetector.detect_non_wear(...)``?" % (
                        wearing_col, wearable.get_pid()))

            wearable.data.loc[wearable.data[wearing_col] == False, self.invalid_col] |= InvCode.FLAG_EPOCH_NON_WEARING

    def flag_day_sleep_length_less_than(self, sleep_period_col: str, min_sleep_in_minutes: int):
        """
        Marks as invalid (InvCode.FLAG_SLEEP) the whole day if the number of slept minutes is smaller than ``min_sleep_in_minutes``.

        :param sleep_period_col:
        :param min_sleep_in_minutes:
        :return:
        """

        for wearable in self.wearables.values():
            if sleep_period_col not in wearable.data.keys():
                warnings.warn("%s is not a valid entry in the data. "
                              "Maybe you need to run ``SleepBoudaryDetector.detect_sleep_boundaries(...)`` first. "
                              "Aborting...." % sleep_period_col)
                return

            sleep_time_per_day = wearable.get_total_sleep_time_per_day(sleep_period_col)
            days_with_problem = sleep_time_per_day < min_sleep_in_minutes
            days_with_problem = days_with_problem[days_with_problem[sleep_period_col] == True].index.values

            if days_with_problem.size > 0:
                wearable.data.loc[
                    wearable.data[wearable.get_experiment_day_col()].isin(
                        days_with_problem), self.invalid_col] |= InvCode.FLAG_DAY_SHORT_SLEEP

    def flag_day_sleep_length_more_than(self, sleep_period_col: str, max_sleep_in_minutes: int):
        """
        Marks as invalid (InvCode.FLAG_LONG_SLEEP) the whole day if the number of slept minutes is larger than ``max_sleep_in_minutes``.
        This analysis is made based on the annotations of the ``sleep_period_col``.

        :param sleep_period_col: Binary sleep column containing: for each epoch, 1 is used for sleep and 0 for awake.
        :param max_sleep_in_minutes: Maximum accepted number of slept minutes per night.
        :return: None
        """

        for wearable in self.wearables.values():
            if sleep_period_col not in wearable.data.keys():
                warnings.warn("%s is not a valid entry in the data. "
                              "Maybe you need to run ``SleepBoudaryDetector.detect_sleep_boundaries(...)`` first. "
                              "Aborting...." % sleep_period_col)
                return

            sleep_time_per_day = wearable.get_total_sleep_time_per_day(sleep_period_col)
            days_with_problem = sleep_time_per_day > max_sleep_in_minutes
            days_with_problem = days_with_problem[days_with_problem[sleep_period_col] == True].index.values

            if days_with_problem.size > 0:
                wearable.data.loc[
                    wearable.data[wearable.get_experiment_day_col()].isin(
                        days_with_problem), self.invalid_col] |= InvCode.FLAG_DAY_LONG_SLEEP

    def _flag_list_OR(self, wearable: Wearable, list_of_flags: list):

        result = wearable.data[self.invalid_col].apply(lambda x: list_of_flags[0] in InvCode.check_flag(x))
        for flag in list_of_flags[1:]:
            result |= wearable.data[self.invalid_col].apply(lambda x: flag in InvCode.check_flag(x))
        return result

    def flag_day_max_nonwearing(self, max_non_wear_minutes_per_day: int):

        for wearable in self.wearables.values():
            epochs_in_minute = wearable.get_epochs_in_min()
            max_non_wear_epochs_per_day = max_non_wear_minutes_per_day * epochs_in_minute

            wearable.data["_tmp_flag_"] = self._flag_list_OR(wearable,
                                                             [InvCode.FLAG_EPOCH_PA, InvCode.FLAG_EPOCH_NON_WEARING,
                                                              InvCode.FLAG_EPOCH_NULL_VALUE])

            invalid_wearing = wearable.data.groupby([wearable.experiment_day_col])["_tmp_flag_"].transform(
                lambda x: x.sum()) >= max_non_wear_epochs_per_day
            wearable.data.loc[invalid_wearing, self.invalid_col] |= InvCode.FLAG_DAY_NON_WEARING
            del wearable.data["_tmp_flag_"]

    def flag_day_if_valid_epochs_smaller_than(self, valid_minutes_per_day: int):

        for wearable in self.wearables.values():
            epochs_in_minute = wearable.get_epochs_in_min()
            valid_epochs_per_day = valid_minutes_per_day * epochs_in_minute

            wearable.data["_tmp_flag_"] = wearable.data[self.invalid_col] == InvCode.FLAG_OKAY

            invalid_epochs = wearable.data.groupby([wearable.experiment_day_col])["_tmp_flag_"].transform(
                lambda x: x.sum()) <= valid_epochs_per_day
            wearable.data.loc[invalid_epochs, self.invalid_col] |= InvCode.FLAG_DAY_NOT_ENOUGH_VALID_EPOCHS
            del wearable.data["_tmp_flag_"]

    def flag_day_without_diary(self):

        for wearable in self.wearables.values():
            tst = wearable.get_total_sleep_time_per_day(based_on_diary=True)
            # Gets the experiment days with 0 total sleep time (i.e., no diary entry)
            invalid_days = set(tst[tst["hyp_diary_sleep"] == 0].index)
            # Flag them as invalid
            if len(invalid_days):
                wearable.data.loc[
                    wearable.data[wearable.get_experiment_day_col()].isin(
                        invalid_days), self.invalid_col] |= InvCode.FLAG_DAY_WITHOUT_DIARY

    def remove_flagged_days(self):
        """
        Fully removes from the wearable data the days that are flagged with problems.
        Only if all epochs in the day are flagged.

        :return: None
        """
        for wearable in self.wearables.values():
            valid_days = self.get_valid_days(wearable.get_pid())[wearable.get_pid()]
            wearable.data = wearable.data[wearable.data[wearable.get_experiment_day_col()].isin(valid_days)].copy()

    def get_invalid_days(self, pid: str = None):
        """
        :return: list of invalid days in the dataset.
        """
        if pid is not None and pid in self.wearables:
            wlist = [self.wearables[pid]]
        else:
            wlist = self.wearables.values()

        r = {}
        for wearable in wlist:
            grp_days = wearable.data.groupby([wearable.get_experiment_day_col()])[self.invalid_col].all().reset_index()
            r[wearable.get_pid()] = set(
                grp_days[grp_days[self.invalid_col] == True][wearable.get_experiment_day_col()].unique())
        return r

    def get_valid_days(self, pid: str = None):
        """
        :return: list of valid days in the dataset.
        """

        if pid is not None and pid in self.wearables:
            wlist = [self.wearables[pid]]
        else:
            wlist = self.wearables.values()

        r = {}
        for wearable in wlist:
            invalid_days = self.get_invalid_days(wearable.get_pid())[wearable.get_pid()]
            all_days = set(wearable.data[wearable.get_experiment_day_col()].unique())
            r[wearable.get_pid()] = all_days - invalid_days
        return r

    def remove_wearable(self, pid):
        if pid in self.wearables:
            del self.wearables[pid]

    def remove_wearables_without_valid_days(self):
        """
        Fully removes any wearable if it does not have any valid data.

        :return: None
        """
        mark_for_removal = []
        for wearable in self.wearables.values():
            valid_days = self.get_valid_days(wearable.get_pid())[wearable.get_pid()]
            if len(valid_days) == 0:
                mark_for_removal.append(wearable.get_pid())

        for pid in mark_for_removal:
            print("Removing wearable %s." % pid)
            self.remove_wearable(pid)

    def flag_day_if_not_enough_consecutive_days(self, min_number_days):

        """
        In case the number of consecutive days is smaller than ``min_number_days``, we mark all as invalid.
        We also try to find any subset that has at least ``min_number_days``.

        :param min_number_days:
        :return:
        """

        for wearable in self.wearables.values():

            days = self.get_valid_days(wearable.get_pid())[wearable.get_pid()]

            if len(days) == 0:
                return

            s = sorted(days)

            consecutive = 1
            last_value = s[0]
            saved_so_far = [last_value]
            okay = []

            for actual in s[1:]:
                if actual == last_value + 1:
                    consecutive += 1
                    last_value = actual
                    saved_so_far.append(last_value)

                else:
                    # Ops! We found a gap in the sequence.
                    # First we check if we already have enough days:
                    if len(saved_so_far) >= min_number_days:
                        okay.extend(saved_so_far)  # Cool! We have enough days.

                    else:  # Otherwise we start over
                        consecutive = 1
                        last_value = actual
                        saved_so_far = [last_value]

            if len(saved_so_far) >= min_number_days:
                okay.extend(saved_so_far)

            # In the okay set, we have all days that we can keep.
            new_invalid = set(days) - set(okay)
            if new_invalid:
                print("Flagging the following days as invalid for pid %s: %s" % (
                    wearable.get_pid(), ','.join(map(str, new_invalid))))

            wearable.data.loc[wearable.data[wearable.get_experiment_day_col()].isin(
                new_invalid), self.invalid_col] |= InvCode.FLAG_DAY_NOT_ENOUGH_CONSECUTIVE_DAYS

    def validation_report(self):
        """
        Generates a report from the actual state of flagged days.
        Days included in this report will be removed if use runs ''remove_flagged_days``.

        :return: None
        """

        day_related_checks = [InvCode.FLAG_DAY_SHORT_SLEEP, InvCode.FLAG_DAY_LONG_SLEEP, InvCode.FLAG_DAY_WITHOUT_DIARY,
                              InvCode.FLAG_DAY_NON_WEARING, InvCode.FLAG_DAY_NOT_ENOUGH_VALID_EPOCHS,
                              InvCode.FLAG_DAY_NOT_ENOUGH_CONSECUTIVE_DAYS]

        total_days = 0
        for check in day_related_checks:
            n_days_check_failed = 0
            for wearable in self.wearables.values():
                wearable.data["_tmp_flag_"] = self._flag_list_OR(wearable, [check])
                n_days_check_failed += wearable.data.groupby([wearable.experiment_day_col])["_tmp_flag_"].all().sum()
            total_days += n_days_check_failed
            print("Number of days removed due to %s: %d" % (check, n_days_check_failed))

        print("Total number of potential days to remove (may have overlaps): %d" % total_days)
