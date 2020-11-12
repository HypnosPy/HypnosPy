from hypnospy import Wearable
from hypnospy import Experiment
from hypnospy import misc
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings

class NonWearingDetector(object):

    def __init__(self, input: {Wearable, Experiment}):
        """ Here we need to load the data and determine:
            potentially by fetching from class wearable
            (1) what type of file is it
            (2) is it multimodal
            (3) length/type- night only/ full
            (4) sampling rate
            """

        if input is None:
            raise ValueError("Invalid value for input.")
        elif type(input) is Wearable:
            self.wearables = [input]
        elif type(input) is Experiment:
            self.wearables = input.get_all_wearables()

        # Those are the new cols that this module is going to generate
        self.wearing_col = "hyp_wearing"  # after running detect
        self.annotation_col = "hyp_annotation"  # 1 for sleep, 0 for awake

        self.sleep_period_col = "hyp_sleep_period"  # after running detect_sleep_boundaries


    def invalidate_day_if_no_sleep(self, sleep_period_col):

        for wearable in self.wearables:
            if sleep_period_col not in wearable.data.keys():
                warnings.warn("%s is not a valid entry in the data. "
                              "Maybe you need to run ``detect_sleep_boundaries(...)`` first. Aborting...." % (
                                  sleep_period_col))
                return
            to_invalidate = wearable.data[sleep_period_col] == -1
            if to_invalidate.sum() > 0:
                print("Invalidating %d rows for pid %s." % (to_invalidate.sum(), wearable.get_pid()))
            wearable.data.loc[to_invalidate, wearable.invalid_col] = True

    def invalidate_day_if_sleep_smaller_than_X_hours(self, X):
        # TODO: missing
        # self.wearable.data.loc[self.wearable.data[self.sleep_period_col] == -1, "hyp_invalid"] = True
        warnings.warn("Missing implementation.")

    def fill_no_activity(self, value):
        for wearable in self.wearables:
            wearable.fill_no_activity(value)

    def detect_non_wear(self,
                        strategy,
                        activity_threshold=0,
                        min_period_len_minutes=90,
                        spike_tolerance=2,
                        min_window_len_minutes=30,
                        window_spike_tolerance=0,
                        use_vector_magnitude=False,
                        ):

        for wearable in self.wearables:
            if strategy in ["choi", "choi2011", "choi11"]:
                if wearable.has_no_activity():
                    # TODO: another way to deal with it is marking those as invalid right away
                    warnings.warn(
                        "It seems pid %s has removed their device. Filling no activity with -0.0001." % wearable.get_pid())
                    wearable.fill_no_activity(-0.0001)

                self.__choi_2011(wearable, activity_threshold, min_period_len_minutes, spike_tolerance,
                                 min_window_len_minutes, window_spike_tolerance, use_vector_magnitude)
            elif strategy in ["none"]:
                wearable.data["%s" % self.wearing_col] = True

            else:
                raise ValueError("Strategy %s not implemented yet." % strategy)

    def __choi_2011(self,
                    wearable,
                    activity_threshold=0,
                    min_period_len_minutes=90,
                    spike_tolerance=2,
                    min_window_len_minutes=30,
                    window_spike_tolerance=0,
                    use_vector_magnitude=False,
                    ):
        """
        Current implementation is largely inspired by shaheen-syed:
        Originally from https://github.com/shaheen-syed/ActiGraph-ActiWave-Analysis/blob/master/algorithms/non_wear_time/choi_2011.py

        Estimate non-wear time based on Choi 2011 paper:
        Med Sci Sports Exerc. 2011 Feb;43(2):357-64. doi: 10.1249/MSS.0b013e3181ed61a3.
        Validation of accelerometer wear and nonwear time classification algorithm.
        Leena Choi, Zhouwen Liu, Charles Matthews, Maciej Buchowski.
        Description from the paper:
        1-min time intervals with consecutive zero counts for at least 90-min time window (window 1), allowing a short time intervals with nonzero counts lasting up to 2 min (allowance interval)
        if no counts are detected during both the 30 min (window 2) of upstream and downstream from that interval; any nonzero counts except the allowed short interval are considered as wearing
        Parameters
        ------------
        activity_threshold : int (optional)
            The activity threshold is the value of the count that is considered "zero", since we are searching for a sequence of zero counts. Default threshold is 0

        spike_tolerance : int (optional)
            Missing

        min_period_len_minutes : int (optional)
            The minimum length of the consecutive zeros that can be considered valid non wear time. Default value is 90 (since we have 60s epoch act_data, this equals 90 mins)
        window_spike_tolerance : int (optional)
            Any count that is above the activity threshold is considered a spike.
            The tolerence defines the number of spikes that are acceptable within a sequence of zeros.
            The default is 2, meaning that we allow for 2 spikes in the act_data, i.e. artifical movement
        min_window_len_minutes : int (optional)
            minimum length of upstream or downstream time window (referred to as window2 in the paper) for consecutive zero counts required before and after the artifactual movement interval to be considered a nonwear time interval.
        use_vector_magnitude: Boolean (optional)
            if set to true, then use the vector magniturde of X,Y, and Z axis, otherwise, use X-axis only. Default False

        Returns
        ---------
        non_wear_vector : np.array((n_samples, 1))
            numpy array with non wear time encoded as 0, and wear time encoded as 1.
        """

        # TODO: write the use case for triaxial devices.
        act_data = wearable.data[wearable.get_activity_col()]
        epochs_in_min = int(wearable.get_epochs_in_min())

        more_than_2_min = (2 * epochs_in_min) + 1
        min_period_len_minutes = epochs_in_min * min_period_len_minutes
        min_window_len_minutes = epochs_in_min * min_window_len_minutes
        spike_tolerance = epochs_in_min * spike_tolerance

        # check if act_data contains at least min_period_len of act_data
        if len(act_data) < min_period_len_minutes:
            raise ValueError("Epoch act_data contains %d samples, "
                             "which is less than the %s minimum required samples" %
                             (len(act_data), min_period_len_minutes))

        # create non wear vector as numpy array with ones.
        # Now we only need to add the zeros which are the non-wear time segments
        non_wear_vector = np.ones((len(act_data), 1), dtype=np.int16)

        """
            ADJUST THE COUNTS IF NECESSARY
        """

        # TODO: missing!
        # if use vector magnitude is set to True,
        # then calculate the vector magnitude of axis 1, 2, and 3, which are X, Y, and Z
        if use_vector_magnitude:
            # calculate vectore
            # TODO: missing implementation
            pass
            #act_data = calculate_vector_magnitude(act_data, minus_one=False, round_negative_to_zero=False)
        # else:
        # if not set to true, then use axis 1, which is the X-axis, located at index 0
        # act_data = act_data[:,0]

        """
            VARIABLES USED TO KEEP TRACK OF NON WEAR PERIODS
        """

        # indicator for resetting and starting over
        reset = False
        # indicator for stopping the non-wear period
        stopped = False
        # indicator for starting to count the non-wear period
        start = False
        # second window validation
        window_2_invalid = False
        # starting minute for the non-wear period
        start_nw = 0
        # ending minute for the non-wear period
        end_nw = 0
        # counter for the number of minutes with intensity between 1 and 100
        cnt_non_zero = 0
        # keep track of non wear sequences
        ranges = []

        """
            FIND NON WEAR PERIODS IN DATA
        """

        # loop over the act_data
        for paxn in range(0, len(act_data)):

            # get the value
            paxinten = act_data[paxn]

            # reset counters if reset or stopped
            if reset or stopped:
                start_nw = 0
                end_nw = 0
                start = False
                reset = False
                stopped = False
                window_2_invalid = False
                cnt_non_zero = 0

            # the non-wear period starts with a zero count
            if paxinten <= 0 and start == False:
                # assign the starting minute of non-wear
                start_nw = paxn
                # set start boolean to true so we know that we started the period
                start = True

            # only do something when the non-wear period has started
            if start:
                # keep track of the number of minutes with intensity that is not a 'zero' count
                if paxinten > activity_threshold:
                    # increase the spike counter
                    cnt_non_zero += 1

                # when there is a non-zero count, check the upstream and downstream window for counts
                # only when the upstream and downstream window have zero counts, then it is a valid non wear sequence
                if paxinten > 0:
                    # check upstream window if there are counts,
                    # note that we skip the count right after the spike, since we allow for 2 minutes of spikes
                    upstream = act_data[paxn + spike_tolerance: paxn + min_window_len_minutes + 1]

                    # check if upstream has non zero counts, if so, then the window is invalid
                    if (upstream > 0).sum() > window_spike_tolerance:
                        window_2_invalid = True

                    # check downstream window if there are counts, again,
                    # we skip the count right before since we allow for 2 minutes of spikes
                    downstream = act_data[
                                 paxn - min_window_len_minutes if paxn - min_window_len_minutes > 0 else 0: paxn - 1]

                    # check if downstream has non zero counts, if so, then the window is invalid
                    if (downstream > 0).sum() > window_spike_tolerance:
                        window_2_invalid = True

                    # if the second window is invalid, we need to reset the sequence for the next run
                    if window_2_invalid:
                        reset = True

                # reset counter if value is "zero" again
                # if paxinten == 0:
                #     cnt_non_zero = 0
                if paxinten <= activity_threshold:
                    cnt_non_zero = 0

                # the sequence ends when there are 3 consecutive spikes,
                # or an invalid second window (upstream or downstream),
                # or the last value of the sequence
                if cnt_non_zero >= more_than_2_min or window_2_invalid or paxn == len(act_data - 1):
                    # define the end of the period
                    end_nw = paxn

                    # check if the sequence is sufficient in length
                    if len(act_data[start_nw:end_nw]) < min_period_len_minutes:
                        # lenght is not sufficient, so reset values in next run
                        reset = True
                    else:
                        # length of sequence is sufficient, set stopped to True so we save the sequence start and end later on
                        stopped = True

                # if stopped is True, the sequence stopped and is valid to include in the ranges
                if stopped:
                    # add ranges start and end non wear time
                    ranges.append([start_nw, end_nw])

        # convert ranges into non-wear sequence vector
        for row in ranges:
            # set the non wear vector according to start and end
            non_wear_vector[row[0]:row[1]] = 0

        non_wear_vector = non_wear_vector.reshape(-1)

        # TODO: Should we get col name by parameter?
        print("Wearable now has a %s col for the non wear flag" % self.wearing_col)
        wearable.data["%s" % self.wearing_col] = non_wear_vector

    def check_valid_days(self, min_activity_threshold: int = 0, max_non_wear_minutes_per_day: int = 180,
                         check_cols: list = [],
                         check_sleep_period: bool = False, sleep_period_col: str = None, check_diary: bool = False):
        """
            Tasks:
            (1) Mark as invalid epochs in which the activity is smaller than the ``min_activity_threshold``.
            (2) Use ``check_cols`` to invalidate Null elements in these col list.
            (3) Mark as invalid the whole day if the number of invalid minutes is bigger than ``max_non_wear_min_per_day``.
            (4) Mark as invalid days with no sleep period (need to run ``detect_sleep_boundaries`` first).
            (5) Mark as invalid days without diary entry.

        """
        for wearable in self.wearables:
            wearable.data[wearable.invalid_col] = False

            # Task 1: Activity smaller than minimal
            wearable.data[wearable.invalid_col] = wearable.data[wearable.invalid_col].where(
                wearable.data[wearable.get_activity_col()] >= min_activity_threshold, True)

            # Task 2: Check if value of col is Null
            for col in check_cols:
                if col not in wearable.data.keys():
                    raise KeyError("Col %s is not available for PID %s" % (col, wearable.get_pid()))
                wearable.data[wearable.invalid_col] = wearable.data[wearable.invalid_col].where(
                    ~(wearable.data[col].isnull()), True)

            # Task 3: Check non-wear in day
            epochs_in_minute = wearable.get_epochs_in_min()
            max_non_wear_epochs_per_day = max_non_wear_minutes_per_day * epochs_in_minute

            if wearable.get_experiment_day_col() is None:
                # If it was not configured yet, we start the experiment day from midnight.
                wearable.change_start_hour_for_experiment_day(0)

            if self.wearing_col not in wearable.data.keys():
                raise KeyError(
                    "Col %s not found in wearable (pid=%s). Did you forget to run ``detect_non_wear(...)``?" % (
                        self.wearing_col, wearable.get_pid()))

            epochs_in_a_day = (1440 * epochs_in_minute)
            invalid_wearing = wearable.data.groupby([wearable.experiment_day_col])[
                                                      self.wearing_col].transform(
                lambda x: x.sum()) <= (epochs_in_a_day - max_non_wear_epochs_per_day)

            wearable.data[wearable.invalid_col] = wearable.data[wearable.invalid_col] | invalid_wearing

            # Task 4: Check sleep period
            if check_sleep_period:
                if sleep_period_col is None:
                    warnings.warn("Need to specify a column to check if sleep period is valid.")
                self.invalidate_day_if_no_sleep(sleep_period_col)

            # Task 5: Check diary entry
            if check_diary:
                if wearable.diary is not None:
                    wearable.invalidate_days_without_diary()
                else:
                    warnings.warn("No diary for PID %s. All days will become invalid." % (wearable.get_pid()))
                    wearable.invalidate_all()

    def check_consecutive_days(self, min_number_days):
        """
        In case the number of consecutive days is smaller than ``min_number_days``, we mark all as invalid.
        We also try to find any subset that has at least ``min_number_days``.

        :param min_number_days:
        :return:
        """

        # TODO: Found a problem here. If the days are 30, 31, 1, 2, 3 the sorted(days) below fails.
        #  A work around is separating the concept of calendar_day and experiment_day

        for wearable in self.wearables:
            days = wearable.get_valid_days()
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

            # In okay we have all days that we can keep.
            new_invalid = set(days) - set(okay)
            if new_invalid:
                print("Marking the following days as invalid for pid %s: %s" % (
                    wearable.get_pid(), ','.join(map(str, new_invalid))))
            wearable.data.loc[wearable.data[wearable.experiment_day_col].isin(new_invalid), wearable.invalid_col] = True

    def get_valid_days(self):
        return_dict = {}
        for wearable in self.wearables:
            pid = wearable.get_pid()
            days = wearable.get_valid_days()
            return_dict[pid] = days
        return return_dict

    def get_invalid_days(self):
        return_dict = {}
        for wearable in self.wearables:
            pid = wearable.get_pid()
            days = wearable.get_invalid_days()
            if len(days) > 0:
                return_dict[pid] = days
        return return_dict

    def drop_invalid_days(self):
        """
        Removes from the wearable dataframe all days that are marked as invalid.

        :param inplace: if ``True`` removes the invalid days from dataframe silently (Default: ``True``)
        :return: return the list remaining dataframe in case inplace is False
        """

        for wearable in self.wearables:
            wearable.drop_invalid_days()
