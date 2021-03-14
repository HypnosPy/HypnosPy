# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
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
from sklearn.metrics import f1_score, make_scorer

from ML_misc import *
from pycaret.classification import *
from glob import glob

def create_filename(dataset, model_str, target, feature_subset, include_past_ys, n_prev_days, predict_d_plus):
    return "outputs/%s/%s_%s_%s_%s_ipast%s_prev%d_future%d" % (dataset, dataset, model_str,
                                                               target, '-'.join(feature_subset),
                                                               include_past_ys, n_prev_days, predict_d_plus)

def exp_from_filename(filename):
    filename = Path(filename).stem
    dataset, model_str, target, features, ipast, prevX, future = filename.split("_")

    ipast = eval(ipast.split("ipast")[1])
    prevX = int(prevX.split("prev")[1])
    future = int(future.split(".")[0].split("future")[1])
    features = [f.replace("+", "_") for f in features.split("-")]

    return dataset, model_str, target, features, ipast, prevX, future


# +
def load_data(filename, dataset, model_str, target, feature_subset, include_past_ys, n_prev_days,
              predict_d_plus, cv_folds=11):
    predict_pa = False
    y_subset = "sleep_metrics"
    keep_pids = True

    # Removes .pkl from file
    experiment_name = os.path.splitext(filename)[0]
    print("EXPERIMENT_NAME", experiment_name)

    df_per_day, df_per_hour, df_per_pid, df_keys, df_embeddings = get_dataframes(dataset, cv_folds)
    age_col = "sleepage5c" if dataset == "mesa" else "AGE_SUENO"

    print("LOG: dataset (%s), model (%s), target (%s), features (%s), days (%d), include_past (%s), predict_pa (%s)" % (
        dataset, model_str, target, '-'.join(feature_subset), n_prev_days, include_past_ys, predict_pa))

    data = get_data(n_prev_days, predict_pa, include_past_ys,
                    df_per_day, df_per_pid, df_keys, df_embeddings,
                    y_subset=y_subset,
                    x_subsets=feature_subset,
                    y_label=target, keep_pids=keep_pids)

    # df_per_pid["sleep_hours"] = df_per_pid[age_col].apply(cdc)
    # data = pd.merge(data, df_per_pid[["sleep_hours", "pid"]])
    df_per_pid["participant_age"] = df_per_pid[age_col]
    data = pd.merge(data, df_per_pid[["participant_age", "pid"]])

    data = data.fillna(-1)
    data = modify_data_target(data, "participant_age", target, keep_others=True)
    
    # Predicting day + 1, instead of day
    if predict_d_plus > 0:
        y = data[[target, "ml_sequence", "pid"]]
        x = data.drop(columns=[target])
        y["ml_sequence"] = y.groupby(["pid"])["ml_sequence"].apply(lambda value: value - predict_d_plus)
        data = pd.merge(x, y)

    cols_to_remove = ["participant_age"] # , "ml_sequence", "pid"]
    for col in cols_to_remove:
        data = data.drop(columns=col)

    test_data = data[data["fold"] == cv_folds - 1]
    data = data[data["fold"] != cv_folds - 1]

    force_cat, force_num = force_categories(dataset, feature_subset)

    experiment = setup(data=data, test_data=test_data,
                       target=target, session_id=123,
                       normalize=True, transformation=True,
                       fold_strategy="groupkfold", fold_groups="fold",
                       categorical_features=force_cat, numeric_features=force_num,
                       ignore_features=["fold", "pid", "ml_sequence", "sleepEfficiency", "totalSleepTime", "awakening"],
                       silent=True, use_gpu=False
                       )

    y_train = pd.concat((data[["sleepEfficiency", "totalSleepTime", "awakening"]], get_config("y_train")), axis=1)
    y_test = pd.concat((test_data[["sleepEfficiency", "totalSleepTime", "awakening"]], get_config("y_test")), axis=1)
    
    data = pd.concat((data[["pid", "ml_sequence", "fold"]], get_config("X_train")), axis=1)
    test_data = pd.concat((test_data[["pid", "ml_sequence", "fold"]], get_config("X_test")), axis=1)

    return data, test_data, y_train, y_test

def write_training_results(model, experiment_filename, dataset, model_str, target, feature_subset,
                                include_past_ys, n_prev_days, predict_d_plus, cv_folds=11):
    dfresult = pull()
    dfresult["dataset"] = dataset
    dfresult["model"] = model_str
    dfresult["target"] = target
    dfresult["feature_set"] = '_'.join(feature_subset)
    dfresult["day_plus_x"] = predict_d_plus
    dfresult["folds"] = cv_folds
    dfresult["tunner_iterations"] = -1
    dfresult["tunner_early_stopping"] = -1
    dfresult["include_past_ys"] = include_past_ys
    dfresult["predict_pa"] = False
    dfresult["n_prev_days"] = n_prev_days
    dfresult["X_shape"] = get_config("X").shape[0]
    dfresult["y_train_shape"] = get_config("y_train").shape[0]
    dfresult["y_test_shape"] = get_config("y_test").shape[0]

    new_filename = get_trainname(experiment_filename)
    dfresult.to_csv(new_filename)
    print("Saved results to: %s" % new_filename)


def predict_test(model, experiment_filename, force_create_training_results, dataset, model_str,
                 target, feature_subset, include_past_ys, n_prev_days, predict_d_plus, cv_folds=11):
    # Force another run to make sure that we save the best results
    print("Creating final model to save results to disk............")


    if force_create_training_results:
        newmodel = create_model(model["trained_model"])
        write_training_results(newmodel, experiment_filename, dataset, model_str, target, feature_subset,
                                    include_past_ys, n_prev_days, predict_d_plus, cv_folds=11)
        predict_model(newmodel)
    else:
        try:
            predict_model(model)
        except:
            newmodel = create_model(model["trained_model"])
            predict_model(newmodel)
            
    # TODO: add parameters to the saved CSV
    dfresult = pull()
    dfresult["dataset"] = dataset
    dfresult["model"] = model_str
    dfresult["target"] = target
    dfresult["feature_set"] = '_'.join(feature_subset)
    dfresult["day_plus_x"] = predict_d_plus
    dfresult["folds"] = cv_folds
    dfresult["include_past_ys"] = include_past_ys
    dfresult["predict_pa"] = False
    dfresult["n_prev_days"] = n_prev_days
    dfresult["test"] = True
    dfresult["X_shape"] = get_config("X").shape[0]
    dfresult["y_train_shape"] = get_config("y_train").shape[0]
    dfresult["y_test_shape"] = get_config("y_test").shape[0]

    new_filename = get_testname(experiment_filename)
    dfresult.to_csv(new_filename)
    print("Saved results to: %s" % new_filename)


# -
DATASET = "mesa" # sys.argv[1]
force_create_training_results = True

# +
#all_files = ["outputs/mesa/mesa_dummy_sleepEfficiency_bins-stats-bouts-time-cosinor-demo-ae24-ae2880-vae24-vae2880-cvae_ipastFalse_prev0_future0.pkl.gz"]

# +
files = {}

all_features = ["bins", "hourly_bins", "stats", "hourly_stats", "bouts", "hourly_bouts", "time", "cosinor",
                "demo", "ae_bouts", "ae24", "vae24"]

for feature_set in all_features:
    files[feature_set] = "outputs/mcc/%s/%s_dummy_combined_%s_ipastFalse_prev0_future0.pkl.gz" % (DATASET, DATASET, feature_set.replace("_", "+"))

X_train, X_val, X_test = {}, {}, {}
for featset, file in files.items():
    
    if file.endswith(".pkl.gz"):
        file = os.path.splitext(file)[0]

    config = exp_from_filename(file)
    X_train[featset], X_test[featset], y_train, y_test = load_data(file, *config)
    
    idx_train = X_train[featset][X_train[featset]["fold"] != 9].index
    idx_val   = X_train[featset][X_train[featset]["fold"] == 9].index
    
    cols_to_delete = ["pid", "ml_sequence", "fold"]
    for col in cols_to_delete:
        del X_train[featset][col]
        del X_test[featset][col]
        
    
    X_val[featset]     = X_train[featset].loc[idx_val]
    X_train[featset]   = X_train[featset].loc[idx_train]
    
    y_val            = y_train.loc[idx_val]
    y_train          = y_train.loc[idx_train]


# +
X, Y = {}, {}
X["train"] = X_train
Y["train"] = y_train
X["val"] = X_val
Y["val"] = y_val
X["test"] = X_test
Y["test"] = y_test

import pickle 
with open("%s_to_NN.pkl" % (DATASET), "wb") as f:
    pickle.dump((X, Y), f)
# -

