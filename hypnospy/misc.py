import warnings
from datetime import timedelta


def get_consecutive_serie(df_in, col):
    df = df_in.copy()

    df["_lenght"] = 0
    df["_seq_id"] = df.groupby((df[col] != df[col].shift(1).fillna(False)).cumsum()).ngroup()
    df["_lenght"] = df[["_lenght", "_seq_id"]].groupby("_seq_id").transform("count")

    return df["_lenght"], df["_seq_id"]


def find_largest_sequence(df_orig, candidate, output_col, seq_length_col="hyp_seq_length", seq_id_col="hyp_seq_id"):
    df = df_orig.copy()
    df[output_col] = False

    df_candidates = df[df[candidate] == True]

    if df_candidates.empty:
        warnings.warn("Day has no valid elements for column %s." % candidate)
        df[output_col] = -1
        return df[output_col]

    #print(df_candidates[seq_id_col].unique(), df_candidates[seq_length_col].unique())
    # Mark the largest period as "sleep_period"
    largest_seqid = df_candidates.iloc[df_candidates[seq_length_col].argmax()][seq_id_col]
    largest = df_candidates[df_candidates[seq_id_col] == largest_seqid]
    df.loc[largest.index, output_col] = True

    return df[output_col]

def merge_windows(df_orig, time_col, sleep_candidate_col, tolerance_minutes=20):
    """

    :param df_orig:
    :param time_col:
    :param sleep_candidate_col:
    :param tolerance_minutes: Use negative to force merging first and last boards
    :return:
    """

    df = df_orig.copy()
    saved_index = df.reset_index()["index"]
    df = df.set_index(time_col)

    df_candidates = df[df[sleep_candidate_col] == True]

    if df_candidates.shape[0] == 0:
        warnings.warn("Day has no sleep period!")
        return df[sleep_candidate_col], df["hyp_seq_id"], df["hyp_seq_length"]

    if tolerance_minutes > 0:

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

    else:
        df.loc[df_candidates.index[0]:df_candidates.index[-1], "hyp_sleep_candidate"] = True
        df.loc[df_candidates.index[0]:df_candidates.index[-1], "hyp_seq_length"] = df.loc[df_candidates.index[0]:df_candidates.index[-1]].shape[0]
        df.loc[df_candidates.index[0]:df_candidates.index[-1], "hyp_seq_id"] = df.loc[df_candidates.index[0]]["hyp_seq_id"]

    # TODO: should we save the index and restore it?
    df.reset_index(inplace=True)
    df.index = saved_index.values

    return df["hyp_sleep_candidate"], df["hyp_seq_id"], df["hyp_seq_length"]
