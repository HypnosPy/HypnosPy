import warnings


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
        warnings.warn("Day has no sleep period!")
        df[output_col] = -1
        return df[output_col]

    # Mark the largest period as "sleep_period"
    largest_seqid = df_candidates.iloc[df_candidates[seq_length_col].argmax()][seq_id_col]
    largest = df_candidates[df_candidates[seq_id_col] == largest_seqid]
    df.loc[largest.index, output_col] = True

    return df[output_col]
