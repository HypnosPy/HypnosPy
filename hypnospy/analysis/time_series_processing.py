from hypnospy import Wearable
from hypnospy import Experiment
from hypnospy import misc
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings

class TimeSeriesProcessing(object):

    def __init__(self, input: {Wearable, Experiment}):
        """ Here we need to load the data and determine:
            potentially by fetching from class wearable
            (1) what type of file is it
            (2) is it multimodal
            (3) length/type- night only/ full
            (4) sampling rate
            """

        if type(input) is Wearable:
            self.wearables = [input]
        elif type(input) is Experiment:
            self.wearables = input.get_all_wearables()

        # Those are the new cols that this module is going to generate
        self.wearing_col = "hyp_wearing"  # after running detect
        self.annotation_col = "hyp_annotation"  # 1 for sleep, 0 for awake

        self.sleep_period_col = "hyp_sleep_period"  # after running detect_sleep_boundaries

    @staticmethod
    def __merge_windows(df_orig, time_col, sleep_candidate_col, tolerance_minutes=20):

        df = df_orig.copy()
        saved_index = df.reset_index()["index"]
        df = df.set_index(time_col)

        df_candidates = df[df[sleep_candidate_col] == True]

        if df_candidates.shape[0] == 0:
            warnings.warn("Day has no sleep period!")
            return df[sleep_candidate_col], df["hyp_seq_id"], df["hyp_seq_length"]

        # Get the list of all sleep candidates
        all_seq_ids = sorted(df_candidates["hyp_seq_id"].unique())  # What are the possible seq_ids?

        actual_sleep_seg_id = all_seq_ids[0]

        for next_sleep_seg_id in all_seq_ids[1:]:

            actual_segment = df[df["hyp_seq_id"] == actual_sleep_seg_id]
            start_time_actual_seg = actual_segment.index[0]
            end_time_actual_seg = actual_segment.index[-1]

            next_segment = df[df["hyp_seq_id"] == next_sleep_seg_id]
            start_time_next_segment = next_segment.index[0]
            end_time_next_segment = next_segment.index[-1]

            if start_time_next_segment - end_time_actual_seg <= timedelta(minutes=tolerance_minutes):
                # Merges two sleep block
                df.loc[start_time_actual_seg:end_time_next_segment, "hyp_seq_id"] = actual_sleep_seg_id
                df.loc[start_time_actual_seg:end_time_next_segment, "hyp_seq_length"] = df.loc[start_time_actual_seg:end_time_next_segment].shape[0]
                df.loc[start_time_actual_seg:end_time_next_segment, "hyp_sleep_candidate"] = True
            else:
                actual_sleep_seg_id = next_sleep_seg_id

        # TODO: should we save the index and restore it?
        df.reset_index(inplace=True)
        df.index = saved_index.values

        return df["hyp_sleep_candidate"], df["hyp_seq_id"], df["hyp_seq_length"]


    @staticmethod
    def __create_threshold_col_based_on_time(df, time_col: str, hr_col: str, start_time: int, end_time: int,
                                             quantile: float, rolling_win_in_minutes: int,
                                             sleep_only_in_sleep_search_window: bool):

        df_time = df.set_index(time_col).copy()

        # This will get the index of everything outside <start_time, end_time>
        idx = df_time.between_time('%02d:00' % end_time,
                                   '%02d:00' % start_time,
                                   include_start=False,
                                   include_end=False).index
        # We set the hr_col to nan for the time outside the search win in order to find the quantile below ignoring nans
        df_time.loc[idx, hr_col] = np.nan

        quantiles_per_day = df_time[hr_col].resample('24H', base=start_time).quantile(quantile).dropna()
        df_time["hyp_sleep"] = quantiles_per_day
        if quantiles_per_day.index[0] < df_time.index[0]:
            df_time.loc[df_time.index[0],"hyp_sleep"] = quantiles_per_day.iloc[0]

        # We fill the nans in the df_time and copy the result back to the original df
        df_time["hyp_sleep"] = df_time["hyp_sleep"].fillna(method='ffill').fillna(method='bfill')

        # binarize_by_hr_threshold
        df_time["hyp_sleep_bin"] = np.where((df_time[hr_col] - df_time["hyp_sleep"]) > 0, 0, 1)
        df_time["hyp_sleep_bin"] = df_time["hyp_sleep_bin"].rolling(window=rolling_win_in_minutes).median().fillna(
            method='bfill')

        if sleep_only_in_sleep_search_window:
            #  Ignore all sleep candidate period outsite win
            df_time.loc[idx, "hyp_sleep_bin"] = 0

        seq_length, seq_id = misc.get_consecutive_serie(df_time, "hyp_sleep_bin")

        return df_time["hyp_sleep"].values, df_time["hyp_sleep_bin"].values, seq_length.values, seq_id.values

    def __sleep_boundaries_with_hr(self, wearable: Wearable,
                                   output_col: str,
                                   quantile: float = 0.4,
                                   volarity_threshold: int = 5,
                                   rolling_win_in_minutes: int = 5,
                                   sleep_search_window: tuple = (20, 12),
                                   min_window_length_in_minutes: int = 40,
                                   volatility_window_in_minutes: int = 10,
                                   merge_blocks_delta_time_in_min: int = 240,
                                   sleep_only_in_sleep_search_window: bool = False,
                                   only_largest_sleep_period: bool = False,
                                   ):

        if wearable.hr_col is None:
            raise AttributeError("HR is not available for PID %s." % (wearable.get_pid()))

        rolling_win_in_minutes = int(rolling_win_in_minutes * wearable.get_epochs_in_min())
        min_window_length_in_minutes = int(min_window_length_in_minutes * wearable.get_epochs_in_min())
        volatility_window_in_minutes = int(volatility_window_in_minutes * wearable.get_epochs_in_min())

        df = wearable.data.copy()

        df["hyp_sleep"], df["hyp_sleep_bin"], df["hyp_seq_length"], df[
            "hyp_seq_id"] = self.__create_threshold_col_based_on_time(wearable.data, wearable.time_col, wearable.hr_col,
                                                                      sleep_search_window[0], sleep_search_window[1],
                                                                      quantile,
                                                                      rolling_win_in_minutes,
                                                                      sleep_only_in_sleep_search_window)

        df['hyp_sleep_candidate'] = ((df["hyp_sleep_bin"] == 1.0) & (
                df['hyp_seq_length'] > min_window_length_in_minutes)).astype(int)

        df["hyp_sleep_vard"] = df[wearable.hr_col].rolling(volatility_window_in_minutes,
                                                                  center=True).std().fillna(0)

        # Merge two sleep segments if their gap is smaller than X min:
        # sleep_segments = df[df["hyp_sleep_candidate"] == 1]["hyp_seq_id"].unique()
        df["hyp_sleep_candidate"], df["hyp_seq_id"], df["hyp_seq_length"] = self.__merge_windows(df, wearable.time_col,
                                                                                                 "hyp_sleep_candidate",
                                                                                                 tolerance_minutes=merge_blocks_delta_time_in_min)

        df = df.set_index(wearable.time_col)
        new_sleep_segments = df[df["hyp_sleep_candidate"] == 1]["hyp_seq_id"].unique()

        # Check if we can modify the sleep onset/offset
        for sleep_seg_id in new_sleep_segments:
            actual_seg = df[df["hyp_seq_id"] == sleep_seg_id]

            if actual_seg.shape[0] == 0:
                continue

            start_time = actual_seg.index[0]
            end_time = actual_seg.index[-1]

            look_sleep_onset = df[start_time - timedelta(hours=4): start_time + timedelta(minutes=60)]
            look_sleep_offset = df[end_time - timedelta(minutes=1): end_time + timedelta(minutes=120)]

            new_sleep_onset = look_sleep_onset[look_sleep_onset["hyp_sleep_vard"] > volarity_threshold]
            new_sleep_offset = look_sleep_offset[look_sleep_offset["hyp_sleep_vard"] > volarity_threshold]

            new_start = new_sleep_onset.index[-1] if not new_sleep_onset.empty else start_time
            new_end = new_sleep_offset.index[0] if not new_sleep_offset.empty else end_time

            df.loc[new_start:new_end, "hyp_seq_id"] = sleep_seg_id
            df.loc[new_start:new_end, "hyp_sleep_candidate"] = 1

        # Need to reorganize the sequences.
        df["hyp_seq_length"], df["hyp_seq_id"] = misc.get_consecutive_serie(df, "hyp_sleep_candidate")

        # new_sleep_segments = df[df[col_win_night + '_sleep_candidate'] == 1][col_win_night + '_grpid'].unique()
        wearable.data = df.reset_index()

        if only_largest_sleep_period:  # If true, we keep only one sleep period per night.

            saved_hour_start_day = wearable.hour_start_experiment
            wearable.change_start_hour_for_experiment_day(sleep_search_window[0])

            grps = wearable.data.groupby(wearable.experiment_day_col)
            tmp_df = []
            for grp_id, grp_df in grps:
                gdf = grp_df.copy()
                gdf["hyp_seq_length"], gdf["hyp_seq_id"] = misc.get_consecutive_serie(gdf, "hyp_sleep_candidate")
                df_out = misc.find_largest_sequence(gdf, "hyp_sleep_candidate", output_col).replace(-1, False)
                tmp_df.append(df_out)
            wearable.data[output_col] = pd.concat(tmp_df)
            wearable.change_start_hour_for_experiment_day(saved_hour_start_day)
        else:
            # Save final output
            wearable.data[output_col] = False
            wearable.data.loc[wearable.data[(wearable.data["hyp_sleep_candidate"] == 1)].index, output_col] = True

        # Clean up!
        wearable.data.drop(
            columns=["hyp_sleep", "hyp_sleep_candidate", "hyp_seq_id",
                     "hyp_sleep_bin",
                     "hyp_sleep_vard", "hyp_seq_length"], inplace=True)

    def __sleep_boundaries_with_annotations(self,
                                            wearable,
                                            output_col,
                                            annotation_col,
                                            hour_to_start_search=18,
                                            merge_tolerance_in_minutes=20,
                                            ):

        if annotation_col is None and self.annotation_col is None:
            raise KeyError("No annotations column specified for pid %s" % wearable.get_pid())

        if annotation_col and annotation_col not in wearable.data.keys():
            raise KeyError("Col %s is not a valid for pid %s" % (annotation_col, wearable.get_pid()))

        saved_hour_start_day = wearable.hour_start_experiment
        wearable.change_start_hour_for_experiment_day(hour_to_start_search)

        if annotation_col is None:
            annotation_col = self.annotation_col

        wearable.data["hyp_sleep_candidate"] = wearable.data[annotation_col]

        # Annotates the sequences of sleep_candidate
        wearable.data["hyp_seq_length"], wearable.data["hyp_seq_id"] = misc.get_consecutive_serie(wearable.data,
                                                                                                  "hyp_sleep_candidate")

        wearable.data["hyp_sleep_candidate"], wearable.data["hyp_seq_id"], wearable.data[
            "hyp_seq_length"] = self.__merge_windows(wearable.data, wearable.time_col, "hyp_sleep_candidate",
                                                 merge_tolerance_in_minutes)

        grps = wearable.data.groupby(wearable.experiment_day_col)
        tmp_df = []
        for grp_id, grp_df in grps:
            gdf = grp_df.copy()
            gdf["hyp_seq_length"], gdf["hyp_seq_id"] = misc.get_consecutive_serie(gdf, "hyp_sleep_candidate")
            df_out = misc.find_largest_sequence(gdf, "hyp_sleep_candidate", output_col).replace(-1, False)
            tmp_df.append(df_out)
        wearable.data[output_col] = pd.concat(tmp_df)

        del wearable.data["hyp_seq_id"]
        del wearable.data["hyp_seq_length"]
        del wearable.data["hyp_sleep_candidate"]

        wearable.change_start_hour_for_experiment_day(saved_hour_start_day)

    def __sleep_boundaries_with_adapted_van_hees(self, wearable: Wearable, output_col: str,
                                                 start_hour: int = 15,
                                                 cols: list = ["pitch_mean_dw", "roll_mean_dw"],
                                                 use_triaxial_activity=False,
                                                 q_sleep: float = 0.1,
                                                 minimum_len_in_minutes: int = 30,
                                                 merge_tolerance_in_minutes: int = 180,
                                                 factor: int = 15,
                                                 operator: str = "or", # Either 'or' or 'and'
                                                 only_largest_sleep_period: bool = False
        ):

        df_time = wearable.data.copy()
        df_time = df_time.set_index(wearable.time_col)

        five_min = int(5 * wearable.get_epochs_in_min())
        minimum_len_in_minutes = int(minimum_len_in_minutes * wearable.get_epochs_in_min())

        if use_triaxial_activity:
            # Step 1:
            df_time["hyp_rolling_x"] = df_time["hyp_act_x"].rolling("5s").median().fillna(0.0)
            df_time["hyp_rolling_y"] = df_time["hyp_act_y"].rolling("5s").median().fillna(0.0)
            df_time["hyp_rolling_z"] = df_time["hyp_act_z"].rolling("5s").median().fillna(0.0)

            df_time["hyp_act_z"].rolling(five_min).median().fillna(0.0)

            df_time["hyp_angle_z"] = (np.arctan(
                df_time["hyp_rolling_z"] / ((df_time['hyp_rolling_y'] ** 2 + df_time['hyp_rolling_x'] ** 2) ** (
                            1 / 2)))) * 180 / np.pi
            # Step 2:
            df_time["hyp_angle_z"] = df_time["hyp_angle_z"].fillna(0.0)
            # Step 3:
            df_time["hyp_angle_z"] = df_time["hyp_angle_z"].rolling("5s").mean().fillna(0.0)

            cols += ["hyp_angle_z"]

        if operator == "or":
            df_time["hyp_sleep_candidate"] = False
        else:
            df_time["hyp_sleep_candidate"] = True

        for col in cols:
            # Paper's Step 4
            df_time["hyp_" + col + '_diff'] = df_time[col].diff().abs()
            # Paper's Step 5
            df_time["hyp_" + col + '_5mm'] = df_time["hyp_" + col + '_diff'].rolling(five_min).median().fillna(0.0)
            # Paper's Step 6
            quantiles_per_day = df_time["hyp_" + col + '_5mm'].resample('24H', base=start_hour).quantile(q_sleep).dropna()
            df_time["hyp_" + col + '_10pct'] = quantiles_per_day
            if quantiles_per_day.index[0] < df_time.index[0]:
                df_time.loc[df_time.index[0], "hyp_" + col + '_10pct'] = quantiles_per_day.iloc[0]

            df_time["hyp_" + col + '_10pct'] = df_time["hyp_" + col + '_10pct'].fillna(method='ffill').fillna(method='bfill')

            df_time["hyp_" + col + '_bin'] = np.where(
                (df_time["hyp_" + col + '_5mm'] - (df_time["hyp_" + col + '_10pct'] * factor)) > 0, 0, 1)
            df_time["hyp_" + col + '_len'], _ = misc.get_consecutive_serie(df_time, "hyp_" + col + '_bin')

            # Paper's Step 7
            if operator == "or":
                df_time["hyp_sleep_candidate"] = df_time["hyp_sleep_candidate"] | ((df_time["hyp_" + col + '_bin'] == 1.0) & (
                        df_time["hyp_" + col + '_len'] > minimum_len_in_minutes))
            else:
                df_time["hyp_sleep_candidate"] = df_time["hyp_sleep_candidate"] & ((
                            df_time["hyp_" + col + '_bin'] == 1.0) & (
                                                         df_time["hyp_" + col + '_len'] > minimum_len_in_minutes))

        # Gets the largest sleep_candidate per night
        wearable.data = df_time.reset_index()
        #wearable.data[output_col] = wearable.data["hyp_sleep_candidate"]

        wearable.data["hyp_seq_length"], wearable.data["hyp_seq_id"] = misc.get_consecutive_serie(wearable.data,
                                                                                                  "hyp_sleep_candidate")
        # Paper's Step 8
        wearable.data["hyp_sleep_candidate"], wearable.data["hyp_seq_id"], wearable.data[
            "hyp_seq_length"] = self.__merge_windows(wearable.data, wearable.time_col, "hyp_sleep_candidate",
                                                     merge_tolerance_in_minutes)

        # Paper's Step 9
        if only_largest_sleep_period:  # If true, we keep only one sleep period per night.
            saved_hour_start_day = wearable.hour_start_experiment
            wearable.change_start_hour_for_experiment_day(start_hour)
            grps = wearable.data.groupby(wearable.experiment_day_col)
            tmp_df = []
            for grp_id, grp_df in grps:
                gdf = grp_df.copy()
                gdf["hyp_seq_length"], gdf["hyp_seq_id"] = misc.get_consecutive_serie(gdf, "hyp_sleep_candidate")
                df_out = misc.find_largest_sequence(gdf, "hyp_sleep_candidate", output_col).replace(-1, False)
                tmp_df.append(df_out)
            wearable.data[output_col] = pd.concat(tmp_df)
            wearable.change_start_hour_for_experiment_day(saved_hour_start_day)
        else:
            # Save final output
            wearable.data[output_col] = False
            wearable.data.loc[wearable.data[(wearable.data["hyp_sleep_candidate"] == 1)].index, output_col] = True

        # Cleaning up...
        cols_to_drop = ["hyp_sleep_candidate", "hyp_seq_length", "hyp_seq_id"]
        for col in cols:
            cols_to_drop.append("hyp_" + col + '_diff')
            cols_to_drop.append("hyp_" + col + '_5mm')
            cols_to_drop.append("hyp_" + col + '_10pct')
            cols_to_drop.append("hyp_" + col + '_len')

        wearable.data.drop(columns=cols_to_drop, inplace=True)

    def detect_sleep_boundaries(self,
                                strategy: str,
                                output_col: str = "hyp_sleep_period",
                                # Should we transform it into kwargs?
                                # Annotation Parameters
                                annotation_hour_to_start_search: int = 18,
                                annotation_col: str = None,
                                annotation_merge_tolerance_in_minutes: int = 20,
                                # HR Parameters
                                hr_quantile: float = 0.4,
                                hr_volarity_threshold: int = 5,
                                hr_rolling_win_in_minutes: int = 5,
                                hr_sleep_search_window: tuple = (20, 12),
                                hr_min_window_length_in_minutes: int = 40,
                                hr_volatility_window_in_minutes: int = 10,
                                hr_merge_blocks_delta_time_in_min: int = 240,
                                hr_sleep_only_in_sleep_search_window: bool = False,
                                hr_only_largest_sleep_period: bool = False,
                                # Adapted Van Hees parameters
                                vanhees_cols: list = ["pitch_mean_dw", "roll_mean_dw"],
                                vanhees_use_triaxial_activity: bool = False,
                                vanhees_start_hour: int = 15,
                                vanhees_quantile: float = 0.1,
                                vanhees_minimum_len_in_minutes: int = 30,
                                vanhees_merge_tolerance_in_minutes: int = 180,
                                vanhees_only_largest_sleep_period: bool = False,
                                ):
        """
            Detected the sleep boundaries.

            param
            -----

            strategy: "annotation", "hr",

            Creates a new col (output_col, default: 'hyp_sleep_period')
            which has value 1 if inside the sleep boundaries and 0 otherwise.

        """
        # (1) HR sleeping window approach here (SLEEP 2020 paper)
        # (2) expert annotations or PSG
        # (3) Van Hees heuristic method
        # Missing: Crespo (periods of innactivity)

        for wearable in self.wearables:
            # print(wearable.get_pid())

            if wearable.data.shape[0] == 0:
                wearable.data[output_col] = np.nan
                warnings.warn("No data for PID %s. Skipping it." % wearable.get_pid())
                continue

            if strategy == "annotation":
                self.__sleep_boundaries_with_annotations(wearable, output_col, annotation_col,
                                                         annotation_hour_to_start_search,
                                                         annotation_merge_tolerance_in_minutes)
            elif strategy == "hr":
                self.__sleep_boundaries_with_hr(wearable, output_col, hr_quantile, hr_volarity_threshold,
                                                hr_rolling_win_in_minutes,
                                                hr_sleep_search_window, hr_min_window_length_in_minutes,
                                                hr_volatility_window_in_minutes, hr_merge_blocks_delta_time_in_min,
                                                hr_sleep_only_in_sleep_search_window, hr_only_largest_sleep_period)

            elif strategy == "adapted_van_hees":
                self.__sleep_boundaries_with_adapted_van_hees(wearable,
                                                              output_col=output_col,
                                                              cols=vanhees_cols,
                                                              use_triaxial_activity=vanhees_use_triaxial_activity,
                                                              start_hour=vanhees_start_hour,
                                                              q_sleep=vanhees_quantile,
                                                              minimum_len_in_minutes=vanhees_minimum_len_in_minutes,
                                                              merge_tolerance_in_minutes=vanhees_merge_tolerance_in_minutes,
                                                              only_largest_sleep_period=vanhees_only_largest_sleep_period,
                                                              )



            else:
                warnings.warn("Strategy %s is not yet implemented" % (strategy))

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
                        "It seems pid %s has removed their device. We will run ``fill_no_activity(-0.0001)`` here.")
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
        Choi L1, Liu Z, Matthews CE, Buchowski MS.
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
            act_data = calculate_vector_magnitude(act_data, minus_one=False, round_negative_to_zero=False)
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
                         check_sleep_period: bool = True, sleep_period_col: str = None, check_diary: bool = True):
        """
            Tasks:
            (1) Mark as invalid epochs in which the activity is smaller than the ``min_activity_threshold``.
            (2) Mark as invalid the whole day if the number of invalid minutes is bigger than ``max_non_wear_min_per_day``.
            (3) Mark as invalid days with no sleep period (need to run ``detect_sleep_boundaries`` first)

        """
        for wearable in self.wearables:
            wearable.data[wearable.invalid_col] = False

            # Task 1:
            wearable.data[wearable.invalid_col] = wearable.data[wearable.invalid_col].where(
                wearable.data[wearable.get_activity_col()] >= min_activity_threshold, True)

            for col in check_cols:
                if col not in wearable.data.keys():
                    raise KeyError("Col %s is not available for PID %s" % (col, wearable.get_pid()))
                wearable.data[wearable.invalid_col] = wearable.data[wearable.invalid_col].where(
                    ~(wearable.data[col].isnull()), True)

            # Task 2:
            freq_in_secs = wearable.get_frequency_in_secs()
            epochs_in_minute = wearable.get_epochs_in_min()

            minutes_in_a_day = 1440 * epochs_in_minute
            max_non_wear_minutes_per_day *= epochs_in_minute

            if wearable.experiment_day_col not in wearable.data.keys():
                # If it was not configured yet, we start the experiment day from midnight.
                wearable.change_start_hour_for_experiment_day(0)

            if self.wearing_col not in wearable.data.keys():
                raise KeyError(
                    "Col %s not found in wearable (pid=%s). Did you forget to run ``detect_non_wear(...)``?" % (
                        self.wearing_col, wearable.get_pid()))

            min_non_wear_min_to_invalid = minutes_in_a_day - max_non_wear_minutes_per_day
            invalid_wearing = wearable.data.groupby([wearable.experiment_day_col])[
                                                      self.wearing_col].transform(
                lambda x: x.sum()) <= min_non_wear_min_to_invalid

            wearable.data[wearable.invalid_col] = wearable.data[wearable.invalid_col] | invalid_wearing

            # Task 3:
            if check_sleep_period:
                if sleep_period_col is None:
                    warnings.warn("Need to specify a column to check if sleep period is valid.")
                self.invalidate_day_if_no_sleep(sleep_period_col)

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
