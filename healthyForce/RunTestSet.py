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
    features = features.split("-")

    return dataset, model_str, target, features, ipast, prevX, future


# +

def load_exp(filename, dataset, model_str, target, feature_subset, include_past_ys, n_prev_days, predict_d_plus,
             cv_folds=11):
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
                    df_per_day, df_per_hour, df_per_pid, df_keys, df_embeddings,
                    y_subset=y_subset,
                    x_subsets=feature_subset,
                    y_label=target, keep_pids=keep_pids)

    # df_per_pid["sleep_hours"] = df_per_pid[age_col].apply(cdc)
    # data = pd.merge(data, df_per_pid[["sleep_hours", "pid"]])
    df_per_pid["participant_age"] = df_per_pid[age_col]
    data = pd.merge(data, df_per_pid[["participant_age", "pid"]])

    data = data.fillna(-1)
    data = modify_data_target(data, "participant_age", target)

    # Predicting day + 1, instead of day
    if predict_d_plus > 0:
        y = data[[target, "ml_sequence", "pid"]]
        x = data.drop(columns=[target])
        y["ml_sequence"] = y.groupby(["pid"])["ml_sequence"].apply(lambda x: x - predict_d_plus)
        data = pd.merge(x, y)

    cols_to_remove = ["ml_sequence", "pid", "participant_age"] # , "sleep_hours"]
    for col in cols_to_remove:
        data = data.drop(columns=col)

    test_data = data[data["fold"] == cv_folds - 1]
    data = data[data["fold"] != cv_folds - 1]

    force_cat, force_num = force_categories(dataset, feature_subset)

    experiment = setup(data=data, test_data=test_data,
                       target=target, session_id=123,
                       normalize=True,
                       transformation=True,
                       fold_strategy="groupkfold",
                       fold_groups="fold",
                       categorical_features=force_cat,
                       numeric_features=force_num,
                       ignore_features=["fold"],
                       silent=True,
                       use_gpu=False
                       )
        
    make_scorer(f1_score, average="macro")
    make_scorer(f1_score, average="micro")
    add_metric(id='micro_f1', name="Micro F1", score_func=lambda x, y: f1_score(x, y, average="macro"),
               greater_is_better=True)
    add_metric(id='macro_f1', name="Macro F1", score_func=lambda x, y: f1_score(x, y, average="micro"),
               greater_is_better=True)
    
    # Metrics removed as it results in problem when using multiclass
    remove_metric('precision')
    remove_metric('recall')
    remove_metric('f1')

    unzip_pkl(experiment_name)
    loaded_model = load_model(experiment_name)
    delete_pkl(experiment_name)

    return loaded_model

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
DATASET = sys.argv[1]
force_create_training_results = True

all_files = glob("outputs/%s/*.pkl.gz" % DATASET)
all_files = [file for file in all_files if "test" not in file]
# all_files = ["outputs/test/mesa_lightgbm_sleepEfficiency_bins-stats-bouts-cosinor_ipastFalse_prev0_future0.pkl.gz"]

for file in tqdm(all_files):
    testname = get_testname(file)
    print(testname)
    if os.path.exists(testname):
        continue

    if file.endswith(".pkl.gz"):
        file = os.path.splitext(file)[0]

    print("ROOT: ", file)

    config = exp_from_filename(file)
    model = load_exp(file, *config)
    predict_test(model, file, force_create_training_results, *config)


