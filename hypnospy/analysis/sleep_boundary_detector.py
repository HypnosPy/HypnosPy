from hypnospy import Wearable
from hypnospy import Experiment
from hypnospy import misc
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings


class SleepBoudaryDetector(object):
    """
    
    Class used to label sleep/wake periods for data in a Wearable object. See HypnosPy book for use example.
    
    """

    def __init__(self, input: {Wearable, Experiment}):

        if type(input) is Wearable:
            self.wearables = [input]
        elif type(input) is Experiment:
            self.wearables = input.get_all_wearables()

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

        quantiles_per_day = df_time[hr_col].resample('24H', offset="%dh" % start_time).quantile(quantile).dropna()
        df_time["hyp_sleep"] = quantiles_per_day
        if quantiles_per_day.index[0] < df_time.index[0]:
            df_time.loc[df_time.index[0], "hyp_sleep"] = quantiles_per_day.iloc[0]

        # We fill the nans in the df_time and copy the result back to the original df
        df_time["hyp_sleep"] = df_time["hyp_sleep"].fillna(method='ffill').fillna(method='bfill')

        # binarize_by_hr_threshold
        df_time["hyp_sleep_bin"] = np.where((df_time[hr_col] - df_time["hyp_sleep"]) > 0, 0, 1)
        df_time["hyp_sleep_bin"] = df_time["hyp_sleep_bin"].rolling(window=rolling_win_in_minutes).median().fillna(
            method='bfill')

        if sleep_only_in_sleep_search_window:
            #  Ignore all sleep candidate period outsite win
            df_time.loc[idx, "hyp_sleep_bin"] = 0

        seq_length, seq_id = misc.get_consecutive_series(df_time, "hyp_sleep_bin")

        return df_time["hyp_sleep"].values, df_time["hyp_sleep_bin"].values, seq_length.values, seq_id.values

    @staticmethod
    def __sleep_boundaries_with_hr(wearable: Wearable, output_col: str, quantile: float = 0.4,
                                   volarity_threshold: int = 5, rolling_win_in_minutes: int = 5,
                                   sleep_search_window: tuple = (20, 12), min_window_length_in_minutes: int = 40,
                                   volatility_window_in_minutes: int = 10, merge_blocks_gap_time_in_min: int = 240,
                                   sleep_only_in_sleep_search_window: bool = False,
                                   only_largest_sleep_period: bool = False):

        if wearable.hr_col is None:
            raise AttributeError("HR is not available for PID %s." % (wearable.get_pid()))

        rolling_win_in_minutes = int(rolling_win_in_minutes * wearable.get_epochs_in_min())
        min_window_length_in_minutes = int(min_window_length_in_minutes * wearable.get_epochs_in_min())
        volatility_window_in_minutes = int(volatility_window_in_minutes * wearable.get_epochs_in_min())

        df = wearable.data.copy()

        df["hyp_sleep"], df["hyp_sleep_bin"], df["hyp_seq_length"], df[
            "hyp_seq_id"] = SleepBoudaryDetector.__create_threshold_col_based_on_time(wearable.data, wearable.time_col,
                                                                                      wearable.hr_col,
                                                                                      sleep_search_window[0],
                                                                                      sleep_search_window[1],
                                                                                      quantile,
                                                                                      rolling_win_in_minutes,
                                                                                      sleep_only_in_sleep_search_window)

        df['hyp_sleep_candidate'] = ((df["hyp_sleep_bin"] == 1.0) & (
                df['hyp_seq_length'] > min_window_length_in_minutes)).astype(int)

        df["hyp_sleep_vard"] = df[wearable.hr_col].rolling(volatility_window_in_minutes,
                                                           center=True).std().fillna(0)

        df["hyp_seq_length"], df["hyp_seq_id"] = misc.get_consecutive_series(df, "hyp_sleep_candidate")

        # Merge two sleep segments if their gap is smaller than X min (interval per day):
        wearable.data = df
        saved_hour_start_day = wearable.hour_start_experiment
        wearable.change_start_hour_for_experiment_day(sleep_search_window[0])
        grps = wearable.data.groupby(wearable.experiment_day_col)
        tmp_df = []
        for grp_id, grp_df in grps:
            gdf = grp_df.copy()
            gdf["hyp_sleep_candidate"], gdf["hyp_seq_length"], gdf["hyp_seq_id"] = misc.merge_sequences_given_tolerance(
                gdf, wearable.time_col, "hyp_sleep_candidate", tolerance_in_minutes=merge_blocks_gap_time_in_min)

            tmp_df.append(gdf)
        wearable.data = pd.concat(tmp_df)
        wearable.change_start_hour_for_experiment_day(saved_hour_start_day)

        df = wearable.data.set_index(wearable.time_col)
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
            # df.loc[new_start:new_end, "hyp_seq_length"] = df.loc[new_start:new_end].shape[0]
            df.loc[new_start:new_end, "hyp_sleep_candidate"] = 1

        # Need to reorganize the sequences.
        df["hyp_seq_length"], df["hyp_seq_id"] = misc.get_consecutive_series(df, "hyp_sleep_candidate")

        # new_sleep_segments = df[df[col_win_night + '_sleep_candidate'] == 1][col_win_night + '_grpid'].unique()
        wearable.data = df.reset_index()

        if only_largest_sleep_period:  # If true, we keep only one sleep period per night.

            saved_hour_start_day = wearable.hour_start_experiment
            wearable.change_start_hour_for_experiment_day(sleep_search_window[0])

            grps = wearable.data.groupby(wearable.experiment_day_col)
            tmp_df = []
            for grp_id, grp_df in grps:
                gdf = grp_df.copy()
                gdf["hyp_seq_length"], gdf["hyp_seq_id"] = misc.get_consecutive_series(gdf, "hyp_sleep_candidate")
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

    @staticmethod
    def __sleep_boundaries_with_annotations(wearable, output_col, annotation_col, hour_to_start_search=18,
                                            merge_tolerance_in_minutes=20, only_largest_sleep_period=True):

        if annotation_col not in wearable.data.keys():
            raise KeyError("Col %s is not a valid for pid %s" % (annotation_col, wearable.get_pid()))

        saved_hour_start_day = wearable.hour_start_experiment
        wearable.change_start_hour_for_experiment_day(hour_to_start_search)

        wearable.data["hyp_sleep_candidate"] = wearable.data[annotation_col].copy()

        # Annotates the sequences of sleep_candidate
        wearable.data["hyp_seq_length"], wearable.data["hyp_seq_id"] = misc.get_consecutive_series(wearable.data,
                                                                                                   "hyp_sleep_candidate")

        wearable.data["hyp_sleep_candidate"], wearable.data["hyp_seq_length"], wearable.data[
            "hyp_seq_id"] = misc.merge_sequences_given_tolerance(wearable.data, wearable.time_col,
                                                                     "hyp_sleep_candidate", merge_tolerance_in_minutes)

        if only_largest_sleep_period:
            grps = wearable.data.groupby(wearable.experiment_day_col)
            tmp_df = []
            for grp_id, grp_df in grps:
                gdf = grp_df.copy()
                gdf["hyp_seq_length"], gdf["hyp_seq_id"] = misc.get_consecutive_series(gdf, "hyp_sleep_candidate")
                df_out = misc.find_largest_sequence(gdf, "hyp_sleep_candidate", output_col).replace(-1, False)
                tmp_df.append(df_out)
            wearable.data[output_col] = pd.concat(tmp_df)
        else:
            wearable.data[output_col] = False
            wearable.data.loc[wearable.data[(wearable.data["hyp_sleep_candidate"] == 1)].index, output_col] = True

        del wearable.data["hyp_seq_id"]
        del wearable.data["hyp_seq_length"]
        del wearable.data["hyp_sleep_candidate"]

        wearable.change_start_hour_for_experiment_day(saved_hour_start_day)

    @staticmethod
    def __sleep_boundaries_with_angle_change_algorithm(wearable: Wearable, output_col: str,
                                                       start_hour: int = 15,
                                                       cols: list = [],
                                                       use_triaxial_activity=False,
                                                       q_sleep: float = 0.1,
                                                       minimum_len_in_minutes: int = 30,
                                                       merge_tolerance_in_minutes: int = 180,
                                                       factor: int = 15,
                                                       operator: str = "or",  # Either 'or' or 'and'
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
            quantiles_per_day = df_time["hyp_" + col + '_5mm'].resample('24H', offset="%dh" % start_hour).quantile(
                q_sleep).dropna()
            # print(quantiles_per_day)

            df_time["hyp_" + col + '_10pct'] = quantiles_per_day
            if quantiles_per_day.index[0] < df_time.index[0]:
                df_time.loc[df_time.index[0], "hyp_" + col + '_10pct'] = quantiles_per_day.iloc[0]

            df_time["hyp_" + col + '_10pct'] = df_time["hyp_" + col + '_10pct'].fillna(method='ffill').fillna(
                method='bfill')

            df_time["hyp_" + col + '_bin'] = np.where(
                (df_time["hyp_" + col + '_5mm'] - (df_time["hyp_" + col + '_10pct'] * factor)) > 0, 0, 1)
            df_time["hyp_" + col + '_len'], _ = misc.get_consecutive_series(df_time, "hyp_" + col + '_bin')

            # Paper's Step 7
            if operator == "or":
                df_time["hyp_sleep_candidate"] = df_time["hyp_sleep_candidate"] | (
                            (df_time["hyp_" + col + '_bin'] == 1.0) & (
                            df_time["hyp_" + col + '_len'] > minimum_len_in_minutes))
            else:
                df_time["hyp_sleep_candidate"] = df_time["hyp_sleep_candidate"] & \
                                                 ((df_time["hyp_" + col + '_bin'] == 1.0)
                                                  & (df_time["hyp_" + col + '_len'] > minimum_len_in_minutes))

        # Gets the largest sleep_candidate per night
        wearable.data = df_time.reset_index()
        # wearable.data[output_col] = wearable.data["hyp_sleep_candidate"]

        wearable.data["hyp_seq_length"], wearable.data["hyp_seq_id"] = misc.get_consecutive_series(wearable.data,
                                                                                                   "hyp_sleep_candidate")
        # Paper's Step 8
        wearable.data["hyp_sleep_candidate"], wearable.data["hyp_seq_length"], wearable.data[
            "hyp_seq_id"] = misc.merge_sequences_given_tolerance(wearable.data, wearable.time_col,
                                                                     "hyp_sleep_candidate", merge_tolerance_in_minutes)

        # Paper's Step 9
        if only_largest_sleep_period:  # If true, we keep only one sleep period per night.
            saved_hour_start_day = wearable.hour_start_experiment
            wearable.change_start_hour_for_experiment_day(start_hour)
            grps = wearable.data.groupby(wearable.experiment_day_col)
            tmp_df = []
            for grp_id, grp_df in grps:
                gdf = grp_df.copy()
                gdf["hyp_seq_length"], gdf["hyp_seq_id"] = misc.get_consecutive_series(gdf, "hyp_sleep_candidate")
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

    def detect_sleep_boundaries(self, strategy: str, output_col: str = "hyp_sleep_period",
                                annotation_hour_to_start_search: int = 18, annotation_col: str = None,
                                annotation_merge_tolerance_in_minutes: int = 20,
                                annotation_only_largest_sleep_period: bool = True,
                                hr_quantile: float = 0.4, hr_volarity_threshold: int = 5,
                                hr_rolling_win_in_minutes: int = 5, hr_sleep_search_window: tuple = (20, 12),
                                hr_min_window_length_in_minutes: int = 40, hr_volatility_window_in_minutes: int = 10,
                                hr_merge_blocks_gap_time_in_min: int = 240,
                                hr_sleep_only_in_sleep_search_window: bool = False,
                                hr_only_largest_sleep_period: bool = False,
                                angle_cols: list = [],
                                angle_use_triaxial_activity: bool = False, angle_start_hour: int = 15,
                                angle_quantile: float = 0.1, angle_minimum_len_in_minutes: int = 30,
                                angle_merge_tolerance_in_minutes: int = 180,
                                angle_only_largest_sleep_period: bool = True):
        """
        Detects and labels sleep boundaries - calls the static labelling methods for each specific labelling strategy

        Parameters
        ----------
        strategy : str
            DESCRIPTION. "hr", "angle","annotation"
            "hr" - HypnosPy HR-based algorithm    
            "angle" - algorithm based on triaxial accelerometry as adapted from van Hees et al. (2018). See HypnosPy page.
            "annotation" - boundaries from sleep diaries or PSG
        
        output_col : str, optional
            DESCRIPTION. The default is "hyp_sleep_period". This is where the sleep/wake period appears as a bool (sleep=1)
        annotation_hour_to_start_search : int, optional
            DESCRIPTION. The default is 18. Start of each analysis day.
        annotation_col : str, optional
            DESCRIPTION. The default is None. Columns that holds the sleep diary / PSG annotations.
        annotation_merge_tolerance_in_minutes : int, optional
            DESCRIPTION. The default is 20. Sleep periods shorter than this are merged. 
        annotation_only_largest_sleep_period : bool, optional
            DESCRIPTION. The default is True. Only keeps one sleep period per analysis day.
        hr_quantile : float, optional
            DESCRIPTION. The default is 0.4. All epochs with HR < hr_quantile (based on 24-hour ECDF) are initially considered as sleep
        hr_volarity_threshold : int, optional
            DESCRIPTION. The default is 5. HR standard deviation threshold (over hr_volatility_window_in_minutes) 
                        used to determine final sleep/wake boundaries  
        hr_rolling_win_in_minutes : int, optional
            DESCRIPTION. The default is 5. Smoothing window for taking the HR rolling average applying the sleep labelling algorithm.
        hr_sleep_search_window : tuple, optional
            DESCRIPTION. The default is (20, 12). Hour window when sleep is assumed to be present (i.e. 8pm-12 noon default)
        hr_min_window_length_in_minutes : int, optional
            DESCRIPTION. The default is 40. Minimum duration for initial sleep windows to progress to the further stages of the algorithm
        hr_volatility_window_in_minutes : int, optional
            DESCRIPTION. The default is 10. Window over which to take the HR volatility/stdev
        hr_merge_blocks_gap_time_in_min : int, optional
            DESCRIPTION. The default is 240. Maximum gap between consecutive sleep periods that can be merged to give the final sleep periods
        hr_sleep_only_in_sleep_search_window : bool, optional
            DESCRIPTION. The default is False. Look for sleep only in the sleep_search_window or the entire analysis day
        hr_only_largest_sleep_period : bool, optional
            DESCRIPTION. The default is False. If True, the algorithm only keeps the largest detected sleep window in each analysis day
        angle_cols : list, optional
            DESCRIPTION. The default is []. Where the accelerometry data is to be taken from.
        angle_use_triaxial_activity : bool, optional
            DESCRIPTION. The default is False. Is the data triaxial (are there 3 angle_cols to analyse?)
        angle_start_hour : int, optional
            DESCRIPTION. The default is 15. Start of each analysis day.
        angle_quantile : float, optional
            DESCRIPTION. The default is 0.1. Activity threshold (based on 24-hour activity ECDF) under which the epoch is initially labelled as sleep
        angle_minimum_len_in_minutes : int, optional
            DESCRIPTION. The default is 30. Minimum duration for initial sleep windows to progress to the further stages of the algorithm
        angle_merge_tolerance_in_minutes : int, optional
            DESCRIPTION. The default is 180. Maximum gap between consecutive sleep periods that can be merged to give the final sleep periods
        angle_only_largest_sleep_period : bool, optional
            DESCRIPTION. The default is True. If True, the algorithm only keeps the largest detected sleep window in each analysis day

        Returns
        -------
        Original Wearable.data, but with a bool output_col containing the sleep/wake label for each epoch

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
                                                         annotation_merge_tolerance_in_minutes,
                                                         annotation_only_largest_sleep_period)
            elif strategy == "hr":
                self.__sleep_boundaries_with_hr(wearable, output_col, hr_quantile, hr_volarity_threshold,
                                                hr_rolling_win_in_minutes, hr_sleep_search_window,
                                                hr_min_window_length_in_minutes, hr_volatility_window_in_minutes,
                                                hr_merge_blocks_gap_time_in_min, hr_sleep_only_in_sleep_search_window,
                                                hr_only_largest_sleep_period)

            elif strategy.lower() in ["adapted_van_hees", "angle"]:
                self.__sleep_boundaries_with_angle_change_algorithm(wearable,
                                                                    output_col=output_col,
                                                                    cols=angle_cols,
                                                                    use_triaxial_activity=angle_use_triaxial_activity,
                                                                    start_hour=angle_start_hour,
                                                                    q_sleep=angle_quantile,
                                                                    minimum_len_in_minutes=angle_minimum_len_in_minutes,
                                                                    merge_tolerance_in_minutes=angle_merge_tolerance_in_minutes,
                                                                    only_largest_sleep_period=angle_only_largest_sleep_period,
                                                                    )



            else:
                warnings.warn("Strategy %s is not yet implemented" % (strategy))


    def _evaluate_sleep_boundaries_pair(self, ground_truth: str, other: str) -> pd.DataFrame:
        """
        Evaluate sleep labels versus ground truth, producing a df with TST, sleep onset, sleep offset, MSE (mean squared error) and Cohen's kappa

        Parameters
        ----------
        ground_truth : str
            DESCRIPTION. file with annotations (sleep diaries or PSG expeert annotations)
        other : str
            DESCRIPTION.

        Returns
        -------
        df_acc : DataFrame
            DESCRIPTION. Evaluation summary

        """

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

    def evaluate_sleep_boundaries(self, ground_truth: str, others: list) -> pd.DataFrame:
        """
        This method aims to compare two or more SleepBoudaryDetector approaches.

        :param ground_truth: Ground Truth data to be compared with sleep_metric[sleep_wake_col] when sleep_perdiod_col == True
        :param others: a list of  0.
        :param sleep_period_col: Dataframe column for the actual sleep period (see SleepBoundaryDetector module)

        :return: a dataframe with the comparison results. Key differ according to the comparison method used.
        """

        dfs = []
        for other in others:
            dfs.append(self._evaluate_sleep_boundaries_pair(ground_truth, other))

        if len(dfs) == 0:
            return None

        result = dfs[0]
        for other in dfs[1:]:
            common_keys = list(set(result.keys()).intersection(other.keys()))
            result = result.merge(other, on=common_keys)

        return result
