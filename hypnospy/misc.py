import warnings
import pandas as pd
import numpy as np
from datetime import timedelta


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


def find_largest_sequence(df_orig: pd.DataFrame, candidate: str, output_col: str, seq_length_col: str = "hyp_seq_length",
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
                                    seq_id_col: str = "hyp_seq_id", seq_length_col: str = "hyp_seq_length") -> [pd.Series, pd.Series, pd.Series]:
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

            if start_time_next_segment - end_time_actual_seg <= timedelta(minutes=tolerance_in_minutes):
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
