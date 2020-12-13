import warnings
import pandas as pd
import numpy as np
import time
import datetime
from calendar import monthrange


def get_consecutive_series(df_in: pd.DataFrame, col: str) -> [pd.Series, pd.Series]:
    """
    This method aims to count the number of consecutive values in the dataframe ``df_in`` indexed by column ``col``.
    Example:
        df = pd.DataFrame({"value": [True, True, True, False, False]})
        len, ids = get_consecutive_series(df, "value")
        len: 3, 3, 3, 2, 2
        ids: 0, 0, 0, 1, 1

    :param df_in: An input dataframe
    :param col:   A column in ``df_in``
    :return: Two pd.Series, the first with the lenght of each sequence and the second with their ids started with 0.
    """
    df = df_in.copy()

    df["_lenght"] = 0
    df["_seq_id"] = df.groupby((df[col] != df[col].shift(1).fillna(False)).cumsum()).ngroup()
    df["_lenght"] = df[["_lenght", "_seq_id"]].groupby("_seq_id").transform("count")

    return df["_lenght"], df["_seq_id"]


def find_largest_sequence(df_orig: pd.DataFrame, candidate: str, output_col: str,
                          seq_length_col: str = "hyp_seq_length",
                          seq_id_col: str = "hyp_seq_id") -> pd.Series:
    df = df_orig.copy()
    df[output_col] = False

    df_candidates = df[df[candidate] == True]

    if df_candidates.empty:
        warnings.warn("Day has no valid elements for column %s." % candidate)
        df[output_col] = -1
        return df[output_col]

    # print(df_candidates[seq_id_col].unique(), df_candidates[seq_length_col].unique())
    # Mark the largest period as "sleep_period"
    largest_seqid = df_candidates.iloc[df_candidates[seq_length_col].argmax()][seq_id_col]
    largest = df_candidates[df_candidates[seq_id_col] == largest_seqid]
    df.loc[largest.index, output_col] = True

    return df[output_col]


def merge_sequences_given_tolerance(df_orig: pd.DataFrame, time_col: str, col_to_act: str, tolerance_in_minutes: int,
                                    seq_id_col: str = "hyp_seq_id", seq_length_col: str = "hyp_seq_length") -> [
    pd.Series, pd.Series, pd.Series]:
    """
    This method is suppose to be used together with ``get_consecutive_series``.
    Here, we want to merge two sequences of positive values from ``get_consecutive_series`` if they are close enough.
    The time proximity or tolerance is given by the parameter ``tolerance_minutes``.
    Here two sequences of True values separated by a sequence of False values will be merged if the length of the sequence of False values is smaller than the tolerance.

    :param df_in: An input dataframe
    :param time_col: a column representing the timestemp of each epoch. Usually ``hyp_time_col``.
    :param col_to_act: a valid column in ``df_in``.
    :param tolerance_in_minutes: Use negative to force merging everything from the first to the last sequence.
    :param seq_id_col: The sequence_id col from ``get_consecutive_series``.
    :param seq_length_col: The sequence_length col from ``get_consecutive_series``.
    :return: Three pd.Series: (1) the new ``col_to_act`` with values replaced according to the tolerance
                              (2) the length of each sequence in the new ``col_to_act``
                              (3) the ids of each sequence in the new ``col_to_act``
    """

    # We expect that the df_orig will not be indexed by the time_col.
    # We start by indexing it with time_col and saving the original index to be able to revert it at the end.
    df = df_orig.copy()
    saved_index = df.reset_index()["index"]
    df = df.set_index(time_col)

    df_true_seq = df[df[col_to_act] == True]

    if df_true_seq.shape[0] == 0:
        warnings.warn("Could not find any valid sequence. Aborting.")
        return df[col_to_act], df[seq_length_col], df[seq_id_col]

    if tolerance_in_minutes > 0:
        # Get the list of all sleep candidates
        all_seq_ids = sorted(df_true_seq[seq_id_col].unique())  # What are the possible seq_ids?

        actual_sleep_seg_id = all_seq_ids[0]

        for next_sleep_seg_id in all_seq_ids[1:]:

            actual_segment = df[df[seq_id_col] == actual_sleep_seg_id]
            start_time_actual_seg = actual_segment.index[0]
            end_time_actual_seg = actual_segment.index[-1]

            next_segment = df[df[seq_id_col] == next_sleep_seg_id]
            start_time_next_segment = next_segment.index[0]
            end_time_next_segment = next_segment.index[-1]

            if start_time_next_segment - end_time_actual_seg <= datetime.timedelta(minutes=tolerance_in_minutes):
                # Merges two sleep block
                df.loc[start_time_actual_seg:end_time_next_segment, seq_id_col] = actual_sleep_seg_id
                df.loc[start_time_actual_seg:end_time_next_segment, seq_length_col] = \
                    df.loc[start_time_actual_seg:end_time_next_segment].shape[0]
                df.loc[start_time_actual_seg:end_time_next_segment, col_to_act] = True
            else:
                actual_sleep_seg_id = next_sleep_seg_id

    else:
        df.loc[df_true_seq.index[0]:df_true_seq.index[-1], col_to_act] = True
        df.loc[df_true_seq.index[0]:df_true_seq.index[-1], seq_length_col] = \
            df.loc[df_true_seq.index[0]:df_true_seq.index[-1]].shape[0]
        df.loc[df_true_seq.index[0]:df_true_seq.index[-1], seq_id_col] = df.loc[df_true_seq.index[0]][
            "hyp_seq_id"]

    df.reset_index(inplace=True)
    df.index = saved_index.values

    return df[col_to_act].astype(np.bool), df[seq_length_col], df[seq_id_col]


def encode_datetime_to_ml(series, col_name) -> pd.DataFrame:
    """
    This method converts datetime pandas series to machine learning acceptable format. 
    It extracts year, month, day, hour, and minute from the datetime object.
    The method returns a dataframe, as shown in below example.
    Example:
        pd.Series
        2017-01-03   2017-01-03 15:25:00
        2017-01-04   2017-01-04 14:56:00
        2017-01-05   2017-01-05 12:49:00
        2017-01-06   2017-01-06 18:52:00
        2017-01-07   2017-01-07 18:00:00
        2017-01-08   2017-01-08 07:58:00
        Freq: 24H, dtype: datetime64[ns]
    
    Code: encode_datetime_to_ml(series, 'acrophase')

    Output:
       acrophase_year  acrophase_month_sin  acrophase_month_cos  \
        2017-01-03            2017                  0.5             0.866025   
        2017-01-04            2017                  0.5             0.866025   
        2017-01-05            2017                  0.5             0.866025   
        2017-01-06            2017                  0.5             0.866025   
        2017-01-07            2017                  0.5             0.866025   
        2017-01-08            2017                  0.5             0.866025   
        ...

    :param series: An input pandas datetime series 
    :param col_name:   prefix column name for output dataframe
    :return: dataframe
    """
    df = pd.DataFrame()
    df[col_name + '_year'] = series.dt.year

    # retain cyclic nature of time
    df[col_name + '_month_sin'] = np.sin(2 * np.pi * series.dt.month / 12)
    df[col_name + '_month_cos'] = np.cos(2 * np.pi * series.dt.month / 12)

    # some months have 28, 29, 30, and 31 days
    days_in_month = series.apply(lambda x: monthrange(x.year, x.month)[1])
    df[col_name + '_day_sin'] = np.sin(2 * np.pi * series.dt.day / days_in_month)
    df[col_name + '_day_cos'] = np.cos(2 * np.pi * series.dt.day / days_in_month)

    df[col_name + '_hour_sin'] = np.sin(2 * np.pi * series.dt.hour / 24)
    df[col_name + '_hour_cos'] = np.cos(2 * np.pi * series.dt.hour / 24)

    df[col_name + '_minute_sin'] = np.sin(2 * np.pi * series.dt.minute / 60)
    df[col_name + '_minute_cos'] = np.cos(2 * np.pi * series.dt.minute / 60)

    return df


def convert_clock_to_sec_since_midnight(t) -> int:
    """
    Converts clock like time (e.g., HH:MM or HH:MM:SS, such as 09:30 or 21:29:59) to seconds since midnight.

    :param t: a string representing the clock time as HH:MM or HH:MM:SS
    :return: seconds since midnight
    """

    if t in ['L', 'H', 'Z']:
        return np.nan
    elif type(t) is not str:
        return np.nan

    n_colon = len(t.split(":")) - 1
    if n_colon == 1:
        x = time.strptime(t, '%H:%M')
    elif n_colon == 2:
        x = time.strptime(t, '%H:%M:%S')
    else:
        raise ValueError("Number of colon should be either one (HH:MM) or two (HH:MM:SS).")

    # seconds since last 00:00
    return datetime.timedelta(hours=x.tm_hour,
                              minutes=x.tm_min,
                              seconds=x.tm_sec).total_seconds()
