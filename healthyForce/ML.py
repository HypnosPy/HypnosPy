#!/usr/bin/env python
# coding: utf-8
# ---
# %%
# -*- coding: utf-8 -*-
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

# +
import sys
from tqdm import tqdm

from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, make_scorer

from ML_misc import *
from pycaret.classification import *
import multiprocessing as mp

import os
import numpy as np

def get_env_var(varname,default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname,':', var)
    else:
        var = default
        print(varname,':', var,'(Default)')
    return var 

SLURM_JOB_ID            = get_env_var('SLURM_JOB_ID', 0)
SLURM_ARRAY_TASK_ID     = get_env_var('SLURM_ARRAY_TASK_ID', 0)
SLURM_ARRAY_TASK_COUNT  = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)
SLURM_JOB_CPUS_PER_NODE = get_env_var('SLURM_JOB_CPUS_PER_NODE', mp.cpu_count())


# +
# %%
predict_pa = False
keep_pids = True
cv_folds = 11
OPT_METRIC = "mcc"

print("Predicting Sleep metrics")

# Arguments: Model, Dataset, GPU, overwrite, fset_num, reverse
# sys.argv = ['0', 'lda', 'mesa', 'False', 'True', 0, 'True']
#possible_models = ['dummy', 'catboost', 'lr', 'lightgbm', 'xgboost', 'rf', 'et', 'lda']

my_models = [sys.argv[1]]
datasets = [sys.argv[2]]
use_gpu = bool(eval(sys.argv[3]))
overwritting = bool(eval(sys.argv[4]))

fset = sys.argv[5]
fset = fset if fset == "all" else int(fset)



print("Using GPU:", use_gpu)
print("Overwritting results:", overwritting)

if predict_pa:
    feature_subsets = [["sleep_metrics"]]
    y_subset = "bouts"
    targets = ["medium_5", "medium_10", "medium_15", "medium_20"]

else:
    
#     all_feature_subsets = ["bins", "stats", "bouts", "hourly_bins", "hourly_bouts", "houly_stats", "time",
#                            "cosinor", "demo", 'ae24', 'ae2880', 'vae24', 'vae2880', 'cvae']
    feature_subsets = [
        # First Batch:
        ["bins", "stats", "bouts", "time", "cosinor", "demo"],                                                  # 0
        ["bins"],                                                                                               # 1
        ["bins", "hourly_bins"],                                                                                # 2
        ["hourly_bins"],                                                                                        # 3
        ["stats"],                                                                                              # 4
        ["hourly_stats"],                                                                                       # 5
        ["stats", "hourly_stats"],                                                                              # 6
        ["bouts"],                                                                                              # 7
        ["hourly_bouts"],                                                                                       # 8
        ["bouts", "hourly_bouts"],                                                                              # 9
        ["cosinor"],                                                                                            # 10
        ["bins", "hourly_bins", "stats", "hourly_stats", "bouts", "hourly_bouts", "time", "cosinor", "demo"],   # 11
        ["bins", "hourly_bins", "stats", "hourly_stats", "bouts", "hourly_bouts", "time", "cosinor"],           # 12
        ["bins", "hourly_bins", "stats", "hourly_stats", "bouts", "hourly_bouts", "cosinor"],                   # 13
        ["ae_bouts"],                                                                                           # 14
        ["ae24"],                                                                                               # 15
        ["vae24"],                                                                                              # 16
        ["bins", "hourly_bins", "stats", "hourly_stats", "bouts", "hourly_bouts", "time", "cosinor", "demo",
                "ae_bouts", "ae24", "vae24"],                                                                   # 17
        ["bins", "hourly_bins", "stats", "hourly_stats", "bouts", "hourly_bouts", "time", "cosinor", "ae_bouts", "ae24", "vae24"],  # 18 removed demo
        # Up to here, select tset 0 - 13
        # Second Batch:
        # 11
        # ["bins", "stats", "bouts", "time", "cosinor", "demo", 'ae24', 'ae2880', 'vae24', 'vae2880', 'cvae'],
        # 12    , 13        , 14       , 15         , 16
        # ['ae24'], ['ae2880'], ['vae24'], ['vae2880'], ['cvae']
    ]
    y_subset = "sleep_metrics"
    targets = ["sleepEfficiency", "awakening", "totalSleepTime", "combined"]


tunner_iterations = 30
tunner_early_stopping = 5

# ====================================================================================
parameters = []

if fset != "all":
    feature_subsets = feature_subsets[fset:fset+1]

for model_str in my_models:
    for dataset in datasets:
        for day_future in range(0, 3):
            for target in targets:
                for feature_subset in feature_subsets:
                    for n_prev_day in range(0, 3):
                        for include_past_ys in [False, True]:
                            parameters.append([dataset, model_str, target, feature_subset, day_future, n_prev_day, include_past_ys])

# ====================================================================================

# %%

selected_parameters = list(np.array_split(parameters, SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])

for param in tqdm(selected_parameters):

    dataset, model_str, target, feature_subset, predict_d_plus, n_prev_days, include_past_ys = param

    experiment_name = "outputs/%s/%s/%s_%s_%s_%s_ipast%s_prev%d_future%d" % (OPT_METRIC, dataset, dataset, model_str, target, '-'.join(feature_subset).replace("_","+"),
                                                               include_past_ys, n_prev_days, predict_d_plus)
    experiment_filename = "%s.csv.gz" % (experiment_name)

    if os.path.exists(experiment_filename) and overwritting is False:
        print("Experiment filename %s already exists. Skipping this one!" % experiment_filename)
        continue

    df_per_day, df_per_hour, df_per_pid, df_keys, df_embeddings = get_dataframes(dataset, cv_folds)
    age_col = "sleepage5c" if dataset == "mesa" else "AGE_SUENO"

    # Merge X, y
    print("LOG: dataset (%s), model (%s), target (%s), features (%s), days (%d), include_past (%s), predict_pa (%s)" % (dataset, model_str, target, '-'.join(feature_subset), n_prev_days, include_past_ys, predict_pa))
    data = get_data(n_prev_days, predict_pa, include_past_ys,
                    df_per_day, df_per_pid, df_keys, df_embeddings,
                    y_subset=y_subset,
                    x_subsets=feature_subset,
                    y_label=target, keep_pids=keep_pids)

    # df_per_pid["sleep_hours"] = df_per_pid[age_col].apply(cdc)
    # data = pd.merge(data, df_per_pid[["sleep_hours", "pid"]])
    df_per_pid["participant_age"] = df_per_pid[age_col].copy()
    data = pd.merge(data, df_per_pid[["participant_age", "pid"]])

    #handout_test_pids = df_per_day[df_per_day["fold"] == cv_folds-1]["pid"].unique()
    
    data = data.fillna(-1)
    data = modify_data_target(data, "participant_age", target)

    # Predicting day + 1, instead of day
    if predict_d_plus > 0:
        y = data[[target, "ml_sequence", "pid"]]
        x = data.drop(columns=[target])
        y["ml_sequence"] = y.groupby(["pid"])["ml_sequence"].apply(lambda value: value - predict_d_plus)
        data = pd.merge(x, y)

    cols_to_remove = ["ml_sequence", "pid", "participant_age"]  # , "sleep_hours"]
    for col in cols_to_remove:
        data = data.drop(columns=col)

    test_data = data[data["fold"] == cv_folds-1]
    data = data[data["fold"] != cv_folds-1]

    force_cat, force_num = force_categories(dataset, feature_subset)

    experiment = setup(data=data,  test_data=test_data, use_gpu=use_gpu,
                   target=target, session_id=123,
                   normalize=True,
                   transformation=True,
                   fold_strategy="groupkfold",
                   fold_groups="fold",
                   # log_experiment = True,
                   # experiment_name = experiment_name,
                   categorical_features=force_cat,
                   numeric_features=force_num,
                   silent=True
                  )
    print("USING GPU? %s" % (get_config('gpu_param')))
    macro_f1 = make_scorer(f1_score, average="macro")
    micro_f1 = make_scorer(f1_score, average="micro")
    add_metric(id='micro_f1', name="Micro F1", score_func=lambda x,y: f1_score(x, y, average="micro"), greater_is_better=True)
    add_metric(id='macro_f1', name="Macro F1", score_func=lambda x,y: f1_score(x, y, average="macro"), greater_is_better=True)
    # Metrics removed as it results in problem when using multiclass
    remove_metric('precision')
    remove_metric('recall')
    remove_metric('f1')
    # Metrics that are okay to leave:
    #remove_metric('kappa')
    #remove_metric('mcc')
    #remove_metric('auc')

    if model_str == 'dummy':
        tunned_model = create_model(DummyClassifier(strategy="most_frequent"))

    else:
        model = create_model(model_str)
        tunned_model = tune_model(model, n_iter=tunner_iterations,
                                  early_stopping=tunner_early_stopping,
                                  search_library="optuna", choose_better=True, optimize=OPT_METRIC)

    save_model(tunned_model, experiment_name)
    zip_pkl(experiment_name)

    # Force another run to make sure that we save the best results
    print("Creating final model to save results to disk............")
    model = create_model(tunned_model)
    # TODO: add parameters to the saved CSV
    dfresult = pull()
    dfresult["dataset"] = dataset
    dfresult["model"] = model_str
    dfresult["target"] = target
    dfresult["feature_set"] = '_'.join(f.replace("_", "+") for f in feature_subset)
    dfresult["day_plus_x"] = predict_d_plus
    dfresult["folds"] = cv_folds
    dfresult["tunner_iterations"] = tunner_iterations
    dfresult["tunner_early_stopping"] = tunner_early_stopping
    dfresult["include_past_ys"] = include_past_ys
    dfresult["predict_pa"] = predict_pa
    dfresult["n_prev_days"] = n_prev_days
    dfresult["X_shape"] = get_config("X").shape[0]
    dfresult["y_train_shape"] = get_config("y_train").shape[0]
    dfresult["y_test_shape"] = get_config("y_test").shape[0]
    dfresult.to_csv(experiment_filename)
    print("Saved results to: %s" % experiment_filename)

    # Save test results
    predict_model(model)

    dfresult = pull()
    dfresult["dataset"] = dataset
    dfresult["model"] = model_str
    dfresult["target"] = target
    dfresult["feature_set"] = '_'.join(f.replace("_", "+") for f in feature_subset)
    dfresult["day_plus_x"] = predict_d_plus
    dfresult["folds"] = cv_folds
    dfresult["include_past_ys"] = include_past_ys
    dfresult["predict_pa"] = predict_pa
    dfresult["n_prev_days"] = n_prev_days
    dfresult["test"] = True
    dfresult["X_shape"] = get_config("X").shape[0]
    dfresult["y_train_shape"] = get_config("y_train").shape[0]
    dfresult["y_test_shape"] = get_config("y_test").shape[0]

    new_filename = get_testname(experiment_filename)
    dfresult.to_csv(new_filename)
    print("Saved TEST results to: %s" % new_filename)

# %%
# Debug:

# include_past_ys = False
# predict_pa = True
# n_prev_days = 1
# predict_d_plus = 1
# target = "sleepEfficiency"

# df_per_day, df_per_hour, df_per_pid, df_keys, df_embeddings = get_dataframes(dataset, cv_folds)

# data = get_data(n_prev_days, predict_pa, include_past_ys,
#                 df_per_day, df_per_pid, df_keys, df_embeddings,
#                 y_subset=y_subset,
#                 x_subsets = feature_subset,
#                 y_label = target, keep_pids=keep_pids)


