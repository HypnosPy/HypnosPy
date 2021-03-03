# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sys
from ML_misc import *


def transform_into_day(df, cols):
    df_tmp = df[["pid", "ml_sequence", "hyp_time_col", *cols]].drop_duplicates()
    
    if 'bout_length' in cols:
        time_col = ["hyp_time_col", 'bout_length']
        cols = list(set(cols) - set(['bout_length']))
    else:
        time_col = ["hyp_time_col"]
        
    
    df_tmp = df_tmp.pivot(["pid", "ml_sequence"], time_col, cols).fillna(0.0)
    df_tmp.columns = ['_'.join(map(str, c)) for c in df_tmp.columns]
    return df_tmp


# +
df_per_day, df_per_hour, df_per_pid, df_keys, df_embeddings = get_dataframes("hchs", 11)
# Calculate additional values
df_stats_hour = transform_into_day(df_per_hour, ['kurtosis', 'max', 'mean', 'median', 'min', 'nunique', 'skewness', 'std'])
df_bins_hour = transform_into_day(df_per_hour, ['light_bins', 'medium_bins', 'sedentary_bins', 'vigorous_bins'])
df_bouts_hour = transform_into_day(df_per_hour, ['bout_length', 'sedentary_bouts', 'light_bouts', 'medium_bouts', 'vigorous_bouts'])

new_df_per_day = pd.merge(df_per_day, df_stats_hour.reset_index()).merge(df_bins_hour.reset_index()).merge(df_bouts_hour.reset_index())
new_df_per_day.to_csv("HCHS_df_per_day.csv")

# Update df_keys
new_keys = {}
new_keys["hourly_stats"] = list(df_stats_hour.keys())
new_keys["hourly_bins"] = list(df_bins_hour.keys())
new_keys["hourly_bouts"] = list(df_bouts_hour.keys())
new_keys = pd.Series(new_keys).reset_index()
new_keys.columns = ["key", "value"]

pd.concat([df_keys.reset_index(), new_keys]).set_index("key").to_csv("HCHS_keys.csv")


# +
df_per_day, df_per_hour, df_per_pid, df_keys, df_embeddings = get_dataframes("mesa", 11)
# Calculate additional values
df_stats_hour = transform_into_day(df_per_hour, ['kurtosis', 'max', 'mean', 'median', 'min', 'nunique', 'skewness', 'std'])
df_bins_hour = transform_into_day(df_per_hour, ['light_bins', 'medium_bins', 'sedentary_bins', 'vigorous_bins'])
df_bouts_hour = transform_into_day(df_per_hour, ['bout_length', 'sedentary_bouts', 'light_bouts', 'medium_bouts', 'vigorous_bouts'])

new_df_per_day = pd.merge(df_per_day, df_stats_hour.reset_index()).merge(df_bins_hour.reset_index()).merge(df_bouts_hour.reset_index())
new_df_per_day.to_csv("MESA_df_per_day.csv")

# Update df_keys
new_keys = {}
new_keys["hourly_stats"] = list(df_stats_hour.keys())
new_keys["hourly_bins"] = list(df_bins_hour.keys())
new_keys["hourly_bouts"] = list(df_bouts_hour.keys())
new_keys = pd.Series(new_keys).reset_index()
new_keys.columns = ["key", "value"]

pd.concat([df_keys.reset_index(), new_keys]).set_index("key").to_csv("MESA_keys.csv")

# +
# Testing:
# df_per_day, df_per_hour, df_per_pid, df_keys, df_embeddings = get_dataframes("hchs", 11)

# +
# feature_subset = ["stats", "hourly_stats"]
# data = get_data(n_prev_days, predict_pa, include_past_ys,
#                     df_per_day, df_per_pid, df_keys, df_embeddings,
#                     y_subset=y_subset,
#                     x_subsets=feature_subset,
#                     y_label=target, keep_pids=True)
