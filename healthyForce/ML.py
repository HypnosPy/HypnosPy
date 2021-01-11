# -*- coding: utf-8 -*-
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

# +
import pandas as pd
import ast
import os
import sys
from glob import glob
from tqdm import tqdm

from sklearn import metrics
from sklearn import dummy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import f1_score, make_scorer

import scipy
# +
def get_columns(dfkeys, subset):
    return dfkeys.loc[subset]["value"]

def map_id_fold(pid_file, n):

    pids = pd.read_csv(pid_file).values.ravel()
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    mapping = []
    for i, (_, test) in enumerate(kf.split(pids)):
        for pid_index in test:
            mapping.append({'fold':i, 'pid':pids[pid_index]})

    return pd.DataFrame(mapping)

def get_dataframes(dataset, nfolds):

    if dataset == "hchs":
        filenames = {"keys": "hchs/HCHS_day_keys.csv", "pids":"hchs/HCHS_pid.csv",
                     "per_day": "hchs/HCHS_per_day.csv",
                     "per_hour": "hchs/HCHS_per_hour.csv", "per_pid": "hchs/HCHS_per_pid.csv"}
    elif dataset == "mesa":
        filenames = {"keys": "mesa/MESA_day_keys.csv", "pids":"mesa/MESA_pid.csv",
                     "per_day": "mesa/MESA_per_day.csv",
                     "per_hour": "mesa/MESA_per_hour.csv", "per_pid": "mesa/MESA_per_pid.csv"}
    else:
        raise ValueError("No Filename for dataset %s" % dataset)

    df_pid_fold = map_id_fold(filenames["pids"], nfolds)

    df_keys = pd.read_csv(filenames["keys"], converters={"value": ast.literal_eval}).set_index("key")

    # Per day
    df_per_day = pd.read_csv(filenames["per_day"])
    df_per_day = pd.merge(df_pid_fold, df_per_day)

    # Per hour
    df_per_hour = pd.read_csv(filenames["per_hour"])
    df_per_hour = pd.merge(df_pid_fold, df_per_hour)

    # Per pid
    df_per_pid = pd.read_csv(filenames["per_pid"])
    df_per_pid = pd.merge(df_pid_fold, df_per_pid)

    return df_per_day, df_per_hour, df_per_pid, df_keys



# -

def get_xy(df_day, df_hour, df_pid, df_keys, y_subset, x_subsets, keep_cols=[]):

    x, y = {}, {}

    dfday = df_day.copy()

    # Extract "Y"
    y_cols = get_columns(df_keys, y_subset)
    print("Y cols: ", y_cols)
    y = dfday[y_cols + keep_cols]

    for exclude_col in y_cols:
        del dfday[exclude_col]

    # Extract "X"
    x_columns = ["fold"] #["pid", "ml_sequence"]
    print(type(x_subsets), x_subsets)

    for subset in list(set(x_subsets) - set(["demo"])):
        x_columns.extend(get_columns(df_keys, subset))

    if "demo" in x_subsets:
        oldcols = df_per_day.keys()
        dfday = pd.merge(df_pid, dfday)
        newcols = set(dfday.keys()) - set(oldcols)
        x_columns.extend(newcols)

    print("X cols: ", x_columns)
    x = dfday[x_columns + keep_cols]

    return x, y


def get_data(n_prev_days, predict_pa, include_past_ys, df_per_day, df_per_hour, df_per_pid, df_keys,
             y_subset="sleep_metrics",
             x_subsets=["bins", "stats", "bouts", "time", "cosinor", "demo"],
             y_label = "sleepEfficiency", keep_pids=False):

    feat_subsets = x_subsets.copy()

    get_demo = False
    if "demo" in feat_subsets:
        get_demo = True
        feat_subsets.remove("demo")

    Xs, Ys = get_xy(df_per_day, df_per_hour, df_per_pid, df_keys, y_subset=y_subset,
                    x_subsets=feat_subsets, keep_cols=["ml_sequence", "pid"])

    Xs_sorted = Xs.sort_values(["pid", "ml_sequence"])
    Ys_sorted = Ys.sort_values(["pid", "ml_sequence"])

    Xsource = Xs_sorted.copy()
    Ysource = Ys_sorted.copy()

    X_cols_to_shift = sorted(list(set(Xs.keys()) - set(["fold", "pid", "ml_sequence"])))
    Y_cols_to_shift = sorted(list(set(Ys.keys()) - set(["fold", "pid", "ml_sequence"])))

    if predict_pa: # Predicting PA next day, we need to use one past day at least
        n_prev_days += 1

    for shift in range(1, n_prev_days + 1):
        # Shift and merge Xs
        Xs_shifted = Xsource.groupby("pid")[X_cols_to_shift].shift(shift)
        new_names = dict([(col, "%s-%dd" % (col, shift)) for col in X_cols_to_shift])
        Xs_shifted.rename(columns=new_names, inplace=True)
        Xs_sorted = pd.concat((Xs_sorted, Xs_shifted), axis=1)

        # Shift and merge Ys
        if include_past_ys:
            Ys_shifted = Ysource.groupby("pid")[Y_cols_to_shift].shift(shift)
            new_names = dict([(col, "%s-%dd" % (col, shift)) for col in Y_cols_to_shift])
            Ys_shifted.rename(columns=new_names, inplace=True)
            Ys_sorted = pd.concat((Ys_sorted, Ys_shifted), axis=1)

    Xs_sorted = Xs_sorted.dropna(axis=0)
    Ys_sorted = Ys_sorted.dropna(axis=0)

    if get_demo:
        Xdemo, _ = get_xy(df_per_day, df_per_hour, df_per_pid, df_keys, y_subset=y_subset,
                          x_subsets=["demo"], keep_cols=["pid", "ml_sequence"])
        Xs_sorted = pd.merge(Xs_sorted, Xdemo)

    new_Y_cols = sorted(list(set(Ys_sorted.keys()) - set(Ysource.keys())))

    if predict_pa: # Predicting PA next day, we also need to remove the sleep metrics for the current night
        #Ys_sorted["ml_sequence"] = Ys_sorted["ml_sequence"] + 1
        X_cols_to_use = sorted(list(set(Xs_sorted.keys()) - set(Xsource.keys())))
        Xs_sorted = Xs_sorted[[*X_cols_to_use, 'fold', 'ml_sequence', 'pid']]


    if not Ys_sorted[new_Y_cols].empty:
        Xs_sorted = pd.merge(Xs_sorted, Ys_sorted[[*new_Y_cols, 'ml_sequence', 'pid']])

    if y_label is not None:
        data = pd.merge(Xs_sorted, Ys[[y_label, "ml_sequence", "pid"]])
    else:
        data = pd.merge(Xs_sorted, Ys)

    if not keep_pids:
        data.drop(columns=["ml_sequence", "pid"], inplace=True)

    return data


# +
def sleepEfficiencyMapping(x):
    if x <= 60:
        return 0
    if x <= 90:
        return 1
    else:
        return 2

def tstMapping(x, interval, tol=1):
    minh, maxh = interval

    if x < (minh - tol) or x > (maxh + tol):
        return 0
    elif x < minh or x > maxh:
        return 1
    else:
        return 2

def awakeningMapping(x):
    if x <= 0:
        return 2
    elif x <= 2:
        return 1
    else:
        return 0

def combinedMapping(x):
    if x <= 2:
        return 0
    elif x <= 5:
        return 1
    else:
        return 2

def cdc(age):
#     https://www.cdc.gov/sleep/about_sleep/how_much_sleep.html
#     School Age	6–12 years	9–12 hours per 24 hours2
#     Teen	13–18 years	8–10 hours per 24 hours2
#     Adult	18–60 years	7 or more hours per night3
#     61–64 years	7–9 hours1
#     65 years and older	7–8 hours1
    if age < 6:
        return (10, 14)
    elif age <= 12:
        return (9, 12)
    elif age <= 18:
        return (8, 10)
    elif age <= 60:
        return (7, 12)
    elif age <= 64:
        return (7, 9)
    else:
        return (7, 8)


# +
def modify_data_target(data, target):

    if target == "awakening" or target is None:
        data["awakening"] = data["awakening"].apply(lambda x: awakeningMapping(x))

    if target == "sleepEfficiency" or target is None:
        data["sleepEfficiency"] = data["sleepEfficiency"].apply(lambda x: sleepEfficiencyMapping(x))

    if target == "totalSleepTime" or target is None:
        data["totalSleepTime"] = data[["totalSleepTime", "sleep_hours"]].apply(lambda x: tstMapping(*x), axis=1)

    if target is None or len(target) == 0:
        data["combined"] = data["totalSleepTime"] + data["sleepEfficiency"] + data["awakening"]
        data["combined"] = data["combined"].apply(lambda x: combinedMapping(x))
        data = data.drop(["sleepEfficiency", "totalSleepTime", "awakening"], axis=1)
        target = "combined"

    return data, target

def force_categories(dataset, feature_subset):
    if "demo" not in feature_subset:
        return [], []

    if dataset == "hchs":
        force_cat = ["FLAG_NARC", 'FLAG_AHIGT50', 'FLAG_AGEGE65', 'GENDERNUM', 'AGEGROUP_C2',
                      'AGEGROUP_C2_SUENO', 'AGEGROUP_C5_SUENO', 'AGEGROUP_C6', 'AGEGROUP_C6_NHANES',
                      'EDUCATION_C2', 'EDUCATION_C3', 'EMPLOYED', 'INCOME', 'INCOME_C3', 'INCOME_C5',
                      'MARITAL_STATUS', 'N_HC', 'OCCUPATION_CURR', 'OCCUPATION_LONG', 'SHIFTWORKERYN',
                      'ALCOHOL_USE_DISORDER', 'ALCOHOL_USE_LEVEL', 'CDCR', 'CDCR_SUENO', 'CHD_SELF',
                      'CHD_SELF_SUENO', 'CIGARETTE_PACK_YEARS_C3', 'CIGARETTE_USE', 'CURRENT_SMOKER',
                      'CLAUDICATION_INT', 'CVD_FRAME', 'DIAB_DIAG', 'DIABETES1', 'DIABETES2', 'DIABETES3',
                      'DIABETES2_INDICATOR', 'DIABETES_C4', 'DIABETES_LAB', 'DIABETES_SELF',
                      'DIABETES_SELF_SUENO', 'DIABETES_SUENO', 'DIAB_FAMHIST', 'DM_AWARE', 'DM_AWARE_SUENO',
                      'DM_CONTROL', 'DOCTOR_VISIT', 'EVER_ANGINA_RELATIVE', 'EVER_CABG_RELATIVE',
                      'EVER_MI_RELATIVE', 'FH_CHD', 'FH_STROKE', 'HYPERT_AWARENESS', 'HYPERT_CONTROL',
                      'HYPERT_TREATMENT', 'HYPERTENSION', 'HYPERTENSION2', 'HYPERTENSION_C4',
                      'HYPERTENSION_SUENO', 'METS_IDF', 'METS_NCEP', 'METS_NCEP2', 'MI_ECG',
                      'PRECHD_ANGINA', 'PRECHD_NO_ANGINA', 'STROKE', 'STROKE_SUENO', 'STROKE_TIA',
                      'STROKE_TIA_SUENO',
                      'SLEA3', 'SLEA4', 'SLEA5', 'SLEA6', 'SLEA7', 'SLEA8', 'SLEA9',
                      'SLEA10', 'SLEA11', 'SLEA12A', 'SLEA12B', 'SLEA12C', 'SLEA12D',
                      'SLEA12E', 'SLEA12F', 'SLEA12G', 'SLEA12H', 'SLEA12I', 'SLEA12J', 'SLEA13', 'SLEA14',
                      'SLEA15', 'SLEA16', 'SLEA17', 'SLEA18',
                      'SPEA10', 'SPEA11', 'SPEA12A', 'SPEA12B', 'SPEA12C', 'SPEA12D', 'SPEA12E', 'SPEA12F',
                      'SPEA12G', 'SPEA12H', 'SPEA12I', 'SPEA12J', 'SPEA13', 'SPEA14', 'SPEA15', 'SPEA16',
                      'SPEA17', 'SPEA18', 'SPEA3', 'SPEA4', 'SPEA5', 'SPEA6', 'SPEA7', 'SPEA8', 'SPEA9', 'SQEA2',
                      'SQEA20', 'SQEA25', 'SQEA3', 'SQEA4', 'SQEA5', 'SQEA6', 'SQEA7', 'SQEA8', 'SQEA9',
                      'SQEA1', 'SQEA10', 'SQEA11', 'SQEA12', 'SQEA13', 'SQEA14', 'SQEA15', 'SQEA16', 'SQEA17',
                      'SQEA18', 'SQEA19',
                     ]
        force_num = ['AGE', 'AGE_SUENO', 'COMMUTEHOME', 'COMMUTEWORK', 'SHIFT_LENGTH', 'TOTCOMMUTE_DAY',
                      'TOTCOMMUTE_WEEK', 'WORK_HRS_DAY', 'WORK_HRS_WEEK', 'CIGARETTE_PACK_YEARS',
                      'CIGARETTES_YEAR', 'EXPOSURE_YEAR', 'FRAME_CVD_RISK_10YR', 'HBA1C_SI', 'HOMA_B',
                      'HOMA_IR', 'TOTALDRINKS_PER_WEEK', 'SQEA21', 'SQEA22', 'SQEA23', 'SQEA24',
                      'SLEA1A_2401', 'SLEA1C_2401', 'SLEA2A_2401', 'SLEA2C_2401']
    else:
        force_cat = ['race1c', 'gender1', 'trbleslpng5', 'bcksleep5', 'wakeup5', 'wakeearly5', 'slpngpills5',
                     'irritable5', 'sleepy5', 'typicalslp5', 'readng5', 'tv5', 'sittng5', 'riding5', 'lyngdwn5',
                     'talkng5', 'quietly5', 'car5', 'dinner5', 'driving5', 'snored5', 'stpbrthng5', 'legsdscmfrt5',
                     'rubbnglgs5', 'wrserest5', 'wrseltr5', 'feelngbstr5', 'tired5', 'mosttired4', 'feelngbstpk5',
                     'types5', 'slpapnea5', 'cpap5', 'dntaldv5', 'uvula5', 'insmnia5', 'rstlesslgs5',
                     'wrksched5', 'extrahrs5']
        force_num = ['sleepage5c', 'wkendsleepdur5t', 'nap5', 'whiirs5c', 'epslpscl5c', 'hoostmeq5c']

    return force_cat, force_num


# +
from pycaret.classification import *

predict_pa = False
keep_pids = True
cv_folds = 11

print("Predicting Sleep metrics")
#possible_models = ['dummy', 'catboost', 'lr', 'lightgbm', 'xgboost', 'rf', 'et', 'lda']
my_models = ["dummy"] # [sys.argv[1]]
datasets = ["hchs"] # [sys.argv[2]]

if predict_pa:
    feature_subsets = [["sleep_metrics"]]
    y_subset = "bouts"
    targets = ["medium_5", "medium_10", "medium_15", "medium_20"]

else:
    # all_feature_subsets = ["bins", "stats", "bouts", "time", "cosinor", "demo"]
    feature_subsets = [
        ["bins", "stats", "bouts", "time", "cosinor", "demo"],
        ["bins", "stats", "bouts"],
        ["bins", "stats", "bouts", "time"],
        ["bins", "stats", "bouts", "cosinor"],
        ["bins", "stats", "bouts", "demo"],
        ["bins", "stats", "bouts", "time", "cosinor"],
        ["bins"], ["stats"], ["bouts"], ["time", "demo"], ["cosinor"]
        ]
    y_subset = "sleep_metrics"
    targets = ["sleepEfficiency", "awakening", "totalSleepTime", None]


tunner_iterations = 50
tunner_early_stopping = 5

# ====================================================================================
parameters = []

for model_str in my_models:
    for dataset in datasets:
        for day_future in range(0, 3):
            for target in targets:
                for feature_subset in feature_subsets:
                    for n_prev_day in range(0, 3):
                        for include_past_ys in [True, False]:
                            parameters.append([dataset, model_str, target, feature_subset, day_future, n_prev_day, include_past_ys])

# ====================================================================================

for param in tqdm(parameters):

    dataset, model_str, target, feature_subset, predict_d_plus, n_prev_days, include_past_ys = param

    experiment_name = "outputs/%s/%s_%s_%s_%s_ipast%s_prev%d_future%d" % (dataset, dataset, model_str, target, '-'.join(feature_subset),
                                                               include_past_ys, n_prev_days, predict_d_plus)
    experiment_filename = "%s.csv.gz" % (experiment_name)

    if os.path.exists(experiment_filename):
        print("Experiment filename %s already exists. Skipping this one!" % (experiment_filename))
        continue

    df_per_day, df_per_hour, df_per_pid, df_keys = get_dataframes(dataset, cv_folds)
    age_col = "sleepage5c" if dataset == "mesa" else "AGE_SUENO"

    print("LOG: dataset (%s), model (%s), target (%s), features (%s), days (%d), include_past (%s), predict_pa (%s)" % (dataset, model_str, target, '-'.join(feature_subset), n_prev_days, include_past_ys, predict_pa))
    data = get_data(n_prev_days, predict_pa, include_past_ys,
                    df_per_day, df_per_hour, df_per_pid, df_keys,
                    y_subset=y_subset,
                    x_subsets = feature_subset,
                    y_label = target, keep_pids=keep_pids)

    df_per_pid["sleep_hours"] = df_per_pid[age_col].apply(cdc)
    data = pd.merge(data, df_per_pid[["sleep_hours", "pid"]])

    #handout_test_pids = df_per_day[df_per_day["fold"] == cv_folds-1]["pid"].unique()
    #handout_test_pids

    data = data.fillna(-1)
    data, target = modify_data_target(data, target)

    # Predicting day + 1, instead of day
    if predict_d_plus > 0:
        y = data[[target, "ml_sequence", "pid"]]
        x = data.drop(columns=[target])
        y["ml_sequence"] = y.groupby(["pid"])["ml_sequence"].apply(lambda x: x + predict_d_plus)
        data = pd.merge(x, y)

    cols_to_remove = ["ml_sequence", "pid", "sleep_hours"]
    for col in cols_to_remove:
        data = data.drop(columns=col)

    test_data = data[data["fold"] == cv_folds-1]
    data = data[data["fold"] != cv_folds-1]

    force_cat, force_num = force_categories(dataset, feature_subset)

    experiment = setup(data = data,  test_data = test_data,
                   target = target, session_id=123,
                   normalize = True,
                   transformation = True,
                   fold_strategy="groupkfold",
                   fold_groups="fold",
                   log_experiment = True,
                   experiment_name = experiment_name,
                   use_gpu=False,
                   categorical_features = force_cat,
                   numeric_features = force_num,
                   silent=True
                  )

    #class_metrics = {}
    #class_metrics["MicroF1"] = make_scorer(f1_score, average="micro")
    #class_metrics["MacroF1"] = make_scorer(f1_score, average="macro")
    macro_f1 = make_scorer(f1_score, average="macro")
    micro_f1 = make_scorer(f1_score, average="micro")
    add_metric(id='micro_f1', name="Micro F1", score_func=lambda x,y: f1_score(x, y, average="macro"), greater_is_better=True)
    add_metric(id='macro_f1', name="Macro F1", score_func=lambda x,y: f1_score(x, y, average="micro"), greater_is_better=True)
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
                                  search_library="optuna", choose_better=True, optimize='micro_f1') #  optimize="F1")

    save_model(tunned_model, experiment_name)
    # Force another run to make sure that we save the best results
    print("Creating final model to save results to disk............")
    model = create_model(tunned_model)
    # TODO: add parameters to the saved CSV
    dfresult = pull()
    dfresult["dataset"] = dataset
    dfresult["model"] = model_str
    dfresult["target"] = target
    dfresult["feature_set"] = '_'.join(feature_subset)
    dfresult["day_plus_x"] = predict_d_plus
    dfresult["folds"] = cv_folds
    dfresult["tunner_iterations"] = tunner_iterations
    dfresult["tunner_early_stopping"] = tunner_early_stopping
    dfresult["include_past_ys"] = include_past_ys
    dfresult["predict_pa"] = predict_pa
    dfresult["n_prev_days"] = n_prev_days
    dfresult.to_csv(experiment_filename)


# +
#model = create_model('catboost')
#predict_model(model)

# Day:
# 0.6700	0.7929	0.6117	0.6801	0.6661	0.4284	0.4316
# 0.4931	0.5000	0.3333	0.2438	0.3261	0.0000	0.0000

# Day + 1:
# 0.5478	0.6649	0.4582	0.5448	0.5337	0.2039	0.2078
# 0.4931	0.5000	0.3333	0.2438	0.3261	0.0000	0.000

#
#print(dataset)

# +
#dummy = create_model(DummyClassifier(strategy="most_frequent"))
#predict_model(dummy)

# +
#evaluate_model(model)
#predict_model(model)
#data

# #setup?
#get_config('y_train')

# +
# OLD CODE WITH REGRESSION
#from pycaret.regression import *
# n_prev_days = 0
# include_past_ys = False
# predict_pa = False

# print("Predicting Sleep metrics")
# my_models = ['dummy', 'catboost', 'br', 'lr', 'lightgbm', 'xgboost', 'et', 'omp']

# if predict_pa:
#     feature_subsets = [["sleep_metrics"]]
#     y_subset = "bouts"
#     targets = ["medium_5", "medium_10", "medium_15", "medium_20"]

# else:
#     # all_feature_subsets = ["bins", "stats", "bouts", "time", "cosinor", "demo"]
#     feature_subsets = [
#             #["bins", "stats", "bouts", "time", "cosinor", "demo"],
#             #       ["bins", "stats", "bouts"],
#             #       ["bins", "stats", "bouts", "time"],
#             #       ["bins", "stats", "bouts", "cosinor"],
#             #       ["bins", "stats", "bouts", "demo"],
#             #       ["bins", "stats", "bouts", "time", "cosinor"],
#             #       ["bins"], ["stats"], ["bouts"], ["time"], ["cosinor"],
#             ["demo"]
#                   ]
#     y_subset = "sleep_metrics"
#     targets = ["sleepEfficiency", "awakening", "totalSleepTime"]


# tunner_iterations = 100
# tunner_early_stopping = 10
# cv_folds = 10

# for dataset in ["mesa"]: # , "hchs"]:
#     df_per_day, df_per_hour, df_per_pid, df_keys = get_dataframes(dataset, cv_folds)

#     for model_str in my_models:
#         for target in targets:
#             for feature_subset in feature_subsets:

#                 print("LOG: dataset (%s), model (%s), target (%s), features (%s), days (%d), include_past (%s), predict_pa (%s)" % (dataset, model_str, target, '-'.join(feature_subset), n_prev_days, include_past_ys, predict_pa))
#                 data = get_data(n_prev_days, predict_pa, include_past_ys,
#                                 df_per_day, df_per_hour, df_per_pid, df_keys,
#                                 y_subset=y_subset,
#                                 x_subsets = feature_subset,
#                                 y_label = target)

#                 experiment_name = "outputs/%s/%s_%s_%s_%s_ipast%s_%d" % (dataset,
#                                                                          dataset,
#                                                                          model_str,
#                                                                          target, '-'.join(feature_subset),
#                                                                          include_past_ys,
#                                                                          n_prev_days
#                                                                          )

#                 experiment = setup(data = data, target = target, session_id=123,
#                                    normalize = True,
#                                    transformation = True,
#                                    fold_strategy="groupkfold",
#                                    fold_groups="fold",
#                                    log_experiment = True,
#                                    experiment_name = experiment_name,
#                                    use_gpu=False,
#                                    silent=True
#                                   )
#                 add_metric(id='pearson', name="Pearson", score_func=lambda a, b: scipy.stats.pearsonr(a,b)[0], greater_is_better=True)

#                 if model_str == 'dummy':
#                     tunned_model = create_model(DummyRegressor())

#                 else:
#                     model = create_model(model_str)
#                     tunned_model = tune_model(model, n_iter=tunner_iterations,
#                                               early_stopping=tunner_early_stopping,
#                                               search_library="optuna", choose_better=True, optimize="R2")

#                 save_model(tunned_model, experiment_name)
#                 # Force another run to make sure that we save the best results
#                 model = create_model(tunned_model)
#                 pull().to_csv(experiment_name + ".csv")


# +
# evaluate_model(best)
# plot_model(best, plot='feature')
# interprete_model(model)

# +
# data[["combined"]]
# comb_pid = data.groupby("pid")["combined"].mean()
# comb_pid.mean()

# #data["combined"].mean()
# #

# #comb_pid.name = "combined_score"
# #diseased_pid = data.groupby("pid")[["rstlesslgs5", "slpapnea5", "insmnia5"]].sum().any(axis=1)
# #diseased_pid.name = "has_disease"

# diseased_pid = data.groupby("pid")[["epslpscl5c", "whiirs5c"]].mean()
# diseased_pid.name = "scores"


# merged = pd.merge(comb_pid, diseased_pid, right_index=True, left_index=True)

# merged.corr("pearson")
# #merged[merged["has_disease"] == True]["combined_score"].mean(), merged[merged["has_disease"] == False]["combined_score"].mean()
# #merged["combined_score"].mean()

# #s = 4
# #merged[merged["combined_score"] > s]["has_disease"].mean(), merged[merged["combined_score"] <= s]["has_disease"].mean()



# +
# #!mlflow ui

# +
#
# dataset = "mesa"
# predict_pa = False
# feature_subset = ["bins", "stats", "bouts", "time", "cosinor", "demo"]
# y_subset = "sleep_metrics"
# y_label = "sleepEfficiency"
# ndays = 2
# include_past_ys = True

# df_per_day, df_per_hour, df_per_pid, df_keys = get_dataframes(dataset, 3)

# data = get_data(ndays, predict_pa, include_past_ys,
#                      df_per_day, df_per_hour, df_per_pid, df_keys,
#                      y_subset=y_subset,
#                      x_subsets = feature_subset,
#                      y_label = y_label)


# +
# dataset = "mesa"
# predict_pa = True
# feature_subset = ['sleep_metrics']
# y_label = "medium_20"
# y_subset = "bouts"
# ndays = 1
# include_past_ys = True

# df_per_day, df_per_hour, df_per_pid, df_keys = get_dataframes(dataset, 3)

# data = get_data(ndays, predict_pa, include_past_ys,
#                      df_per_day, df_per_hour, df_per_pid, df_keys,
#                      y_subset=y_subset,
#                      x_subsets = feature_subset,
#                      y_label = y_label)

# +
#data[["ml_sequence", "pid", "medium_20", "medium_20-1d", "sleepEfficiency", "sleepEfficiency-1d"]].head(10)
#data[["ml_sequence", "pid", "medium_20", "sleepEfficiency",]].head(10)
#data.head(10)

# +
### If we want to run only one experiment
# experiment = setup(data = data, target = y_label, session_id=123,
#                    normalize = True,
#                    transformation = True,
#                    fold_strategy="groupkfold",
#                    fold_groups="fold",
#                    fold=3,
#                    log_experiment = True,
#                    experiment_name = experiment_name,
#                    use_gpu=False,
#                    silent=True
#                   )

# +
# best = compare_models()

# +
# evaluate_model(best)
