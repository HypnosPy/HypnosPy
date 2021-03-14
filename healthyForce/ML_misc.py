# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
import ast
import os
import gzip
import shutil

from sklearn.model_selection import KFold

# +
valid_embedders = ['ae24', 'ae2880', 'vae24', 'vae2880', 'cvae', 'ae_bouts']

def get_columns(dfkeys, subset):
    return dfkeys.loc[subset]["value"]


def map_id_fold(pid_file, n):
    pids = pd.read_csv(pid_file).values.ravel()
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    mapping = []
    for i, (_, test) in enumerate(kf.split(pids)):
        for pid_index in test:
            mapping.append({'fold': i, 'pid': pids[pid_index]})

    return pd.DataFrame(mapping)


def load_embeddings(train, test):
    embeddings_train = pd.read_pickle(train)
    # embeddings_train.columns = embeddings_train.columns.droplevel(1)

    embeddings_test = pd.read_pickle(test)
    # embeddings_test.columns = embeddings_test.columns.droplevel(1)

    df_embeddings = pd.concat([embeddings_train, embeddings_test])

    return df_embeddings


def get_dataframes(dataset, nfolds):
    folder_name = ''
    #         folder_name = 'acm_health_sleep_data-main/'

    if dataset == "hchs":
        filenames = {"keys": folder_name + "processed_hchs/HCHS_day_keys.csv.gz",
                     "pids": folder_name + "processed_hchs/HCHS_pid.csv.gz",
                     "per_day": folder_name + "processed_hchs/HCHS_per_day.csv.gz",
                     "per_hour": folder_name + "processed_hchs/HCHS_per_hour.csv.gz",
                     "per_pid": folder_name + "processed_hchs/HCHS_per_pid.csv.gz",
                     "embeddings_train": folder_name + "processed_hchs/HCHS_embeddings_train.pkl.gz",
                     "embeddings_test": folder_name + "processed_hchs/HCHS_embeddings_test.pkl.gz"
                     }
    elif dataset == "mesa":
        filenames = {"keys": folder_name + "processed_mesa/MESA_day_keys.csv.gz",
                     "pids": folder_name + "processed_mesa/MESA_pid.csv.gz",
                     "per_day": folder_name + "processed_mesa/MESA_per_day.csv.gz",
                     "per_hour": folder_name + "processed_mesa/MESA_per_hour.csv.gz",
                     "per_pid": folder_name + "processed_mesa/MESA_per_pid.csv.gz",
                     "embeddings_train": folder_name + "processed_mesa/MESA_embeddings_train.pkl.gz",
                     "embeddings_test": folder_name + "processed_mesa/MESA_embeddings_test.pkl.gz"
                     }
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

    # Embeddings
    df_embeddings = load_embeddings(filenames['embeddings_train'], filenames['embeddings_test'])
    df_embeddings = df_embeddings.rename(columns={'bout': 'ae_bouts'})
    
    return df_per_day, df_per_hour, df_per_pid, df_keys, df_embeddings


# Filters out embedders from feature sets.
def map_feature_set_to_embedders(x_subsets):
    requested_embedders = []
    for s in x_subsets:
        # .startswith in case of ae24_version2
        if s.startswith(tuple(valid_embedders)):
            requested_embedders.append(s)

    return requested_embedders


def get_xy(df_day, df_pid, df_keys, df_embeddings, y_subset, x_subsets, keep_cols=[]):

    dfday = df_day.copy()

    # Extract "Y"
    y_cols = get_columns(df_keys, y_subset)
    print("Y cols: ", y_cols)
    y = dfday[y_cols + keep_cols]

    for exclude_col in y_cols:
        del dfday[exclude_col]

    # Extract "X"
    x_columns = ["fold"]  # ["pid", "ml_sequence"]
    print(type(x_subsets), x_subsets)

    embedders = map_feature_set_to_embedders(x_subsets)

    for subset in list(set(x_subsets) - set(["demo"]) - set(embedders)):
        x_columns.extend(get_columns(df_keys, subset))

    if "demo" in x_subsets:
        oldcols = df_day.keys()
        dfday = pd.merge(df_pid, dfday)
        newcols = set(dfday.keys()) - set(oldcols)
        x_columns.extend(newcols)

    print("X cols: ", x_columns)
    x = dfday[x_columns + keep_cols]

    # Check if x_subsets contain embedded features
    if embedders:
        # Remove ae2888, vae2888, cvae since their learning wasn't good.
        df_embeddings = df_embeddings[embedders]

        # Give each column a unique feature name (ae24.1, ae24.2, ...)
        df_embeddings.columns = ['.'.join((model, str(feature))).strip() for (model, feature) in
                                 df_embeddings.columns.values]
        x = x.merge(df_embeddings, left_on=['pid', 'ml_sequence'], right_index=True)

    return x, y


def get_data(n_prev_days, predict_pa, include_past_ys, df_per_day, df_per_pid, df_keys,
             df_embeddings,
             y_subset="sleep_metrics",
             x_subsets=["bins", "stats", "bouts", "time", "cosinor", "demo"],
             y_label="sleepEfficiency", keep_pids=False):
    feat_subsets = x_subsets.copy()

    get_demo = False
    if "demo" in feat_subsets:
        get_demo = True
        feat_subsets.remove("demo")

    # TODO: pass embedded_df_per_hour to get_xy
    # gets features and label from df_per_day,
    Xs, Ys = get_xy(df_per_day, df_per_pid, df_keys, df_embeddings, y_subset=y_subset,
                    x_subsets=feat_subsets, keep_cols=["ml_sequence", "pid"])

    Xs_sorted = Xs.sort_values(["pid", "ml_sequence"])
    Ys_sorted = Ys.sort_values(["pid", "ml_sequence"])

    Xsource = Xs_sorted.copy()
    Ysource = Ys_sorted.copy()

    X_cols_to_shift = sorted(list(set(Xs.keys()) - set(["fold", "pid", "ml_sequence"])))
    Y_cols_to_shift = sorted(list(set(Ys.keys()) - set(["fold", "pid", "ml_sequence"])))

    if predict_pa:  # Predicting PA next day, we need to use one past day at least
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
        Xdemo, _ = get_xy(df_per_day, df_per_pid, df_keys, df_embeddings, y_subset=y_subset,
                          x_subsets=["demo"], keep_cols=["pid", "ml_sequence"])

        Xs_sorted = pd.merge(Xs_sorted, Xdemo)

    new_Y_cols = sorted(list(set(Ys_sorted.keys()) - set(Ysource.keys())))

    if predict_pa:  # Predicting PA next day, we also need to remove the sleep metrics for the current night
        # Ys_sorted["ml_sequence"] = Ys_sorted["ml_sequence"] + 1
        X_cols_to_use = sorted(list(set(Xs_sorted.keys()) - set(Xsource.keys())))
        Xs_sorted = Xs_sorted[[*X_cols_to_use, 'fold', 'ml_sequence', 'pid']]

    if not Ys_sorted[new_Y_cols].empty:
        Xs_sorted = pd.merge(Xs_sorted, Ys_sorted[[*new_Y_cols, 'ml_sequence', 'pid']])

    if y_label != "combined":
        data = pd.merge(Xs_sorted, Ys[[y_label, "ml_sequence", "pid"]])
    else:
        data = pd.merge(Xs_sorted, Ys)

    if not keep_pids:
        data.drop(columns=["ml_sequence", "pid"], inplace=True)

    print(Xs_sorted.columns)
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
    elif x <= 4:
        return 1
    else:
        return 2


def get_age_group(age):
    # https://www.sciencedirect.com/science/article/pii/S2352721816301309
    if age <= 1:  # infant
        return 0
    elif age <= 2:  # toddler
        return 1
    elif age <= 5:  # prescholler
        return 2
    elif age <= 13:  # school
        return 3
    elif age <= 17:  # teenager
        return 4
    elif age <= 25:  # young adult
        return 5
    elif age <= 64:  # adult
        return 6
    # older adult
    return 7


def awakenings_by_age(awakenings, age):
    age_grp = get_age_group(age)
    if age_grp in [0, 1, 2, 3, 5, 6]:
        return 2 if awakenings <= 1 else 1 if awakenings <= 3 else 0
    elif age_grp == 4:
        return 2 if awakenings <= 1 else 1 if awakenings <= 2 else 0
    elif age_grp == 7:
        return 2 if awakenings <= 2 else 1 if awakenings <= 3 else 0
    else:
        print("wrong age_grp: %d for age %d" % (age_grp, age))


def sleep_efficiency_by_age(se, age):
    age_grp = get_age_group(age)
    if age_grp in [0, 1, 2, 3, 4, 6, 7]:
        return 2 if se >= 85 else 1 if se >= 75 else 0
    elif age_grp == 5:
        return 2 if se >= 85 else 1 if se >= 65 else 0


def sleep_length_by_age(sl, age):
    # https://www.sciencedirect.com/science/article/pii/S2352721815000157
    age_grp = get_age_group(age)
    if age_grp == 0:  # Infant
        return 2 if 12 <= sl <= 15 else 1 if 10 <= sl <= 18 else 0
    if age_grp == 1:  # toddler
        return 2 if 11 <= sl <= 14 else 1 if 9 <= sl <= 16 else 0
    if age_grp == 2:  # Preschoolers
        return 2 if 10 <= sl <= 13 else 1 if 8 <= sl <= 14 else 0
    if age_grp == 3:  # School-aged children
        return 2 if 9 <= sl <= 11 else 1 if 7 <= sl <= 12 else 0
    if age_grp == 4:  # Teenagers
        return 2 if 8 <= sl <= 10 else 1 if 7 <= sl <= 11 else 0
    if age_grp == 5:  # Young adult
        return 2 if 7 <= sl <= 9 else 1 if 6 <= sl <= 11 else 0
    if age_grp == 6:  # Adults
        return 2 if 7 <= sl <= 9 else 1 if 6 <= sl <= 10 else 0
    if age_grp == 7:  # Older adults
        return 2 if 7 <= sl <= 8 else 1 if 5 <= sl <= 9 else 0


def cdc_sleep_length(age):
    #
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


def modify_data_target(data, age_col, target, keep_others=False):
    if target == "awakening":
        data["awakening"] = data[["awakening", age_col]].apply(lambda x: awakenings_by_age(*x), axis=1)

    elif target == "sleepEfficiency":
        data["sleepEfficiency"] = data[["sleepEfficiency", age_col]].apply(lambda x: sleep_efficiency_by_age(*x), axis=1)

    elif target == "totalSleepTime":
        data["totalSleepTime"] = data[["totalSleepTime", age_col]].apply(lambda x: sleep_length_by_age(*x), axis=1)

    elif target == "combined":
        data["awakening"] = data[["awakening", age_col]].apply(lambda x: awakenings_by_age(*x), axis=1)
        data["sleepEfficiency"] = data[["sleepEfficiency", age_col]].apply(lambda x: sleep_efficiency_by_age(*x), axis=1)
        data["totalSleepTime"] = data[["totalSleepTime", age_col]].apply(lambda x: sleep_length_by_age(*x), axis=1)

        data["combined"] = data["totalSleepTime"] + data["sleepEfficiency"] + data["awakening"]
        data["combined"] = data["combined"].apply(lambda x: combinedMapping(x))
        if not keep_others:
            data = data.drop(["sleepEfficiency", "totalSleepTime", "awakening"], axis=1)

    return data


def force_categories(dataset, feature_subset):
    force_cat, force_num = [], []
    
    if "time" in feature_subset:
        force_cat.append("hyp_weekday")
    
    if "demo" not in feature_subset:
        return force_cat, force_num

    if dataset == "hchs":
        force_cat.extend(["FLAG_NARC", 'FLAG_AHIGT50', 'FLAG_AGEGE65', 'GENDERNUM', 'AGEGROUP_C2',
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
                     'SQEA18', 'SQEA19'])
        force_num.extend(['AGE', 'AGE_SUENO', 'COMMUTEHOME', 'COMMUTEWORK', 'SHIFT_LENGTH', 'TOTCOMMUTE_DAY',
                     'TOTCOMMUTE_WEEK', 'WORK_HRS_DAY', 'WORK_HRS_WEEK', 'CIGARETTE_PACK_YEARS',
                     'CIGARETTES_YEAR', 'EXPOSURE_YEAR', 'FRAME_CVD_RISK_10YR', 'HBA1C_SI', 'HOMA_B',
                     'HOMA_IR', 'TOTALDRINKS_PER_WEEK', 'SQEA21', 'SQEA22', 'SQEA23', 'SQEA24',
                     'SLEA1A_2401', 'SLEA1C_2401', 'SLEA2A_2401', 'SLEA2C_2401'])
    else:
        force_cat.extend(['race1c', 'gender1', 'trbleslpng5', 'bcksleep5', 'wakeup5', 'wakeearly5', 'slpngpills5',
                     'irritable5', 'sleepy5', 'typicalslp5', 'readng5', 'tv5', 'sittng5', 'riding5', 'lyngdwn5',
                     'talkng5', 'quietly5', 'car5', 'dinner5', 'driving5', 'snored5', 'stpbrthng5', 'legsdscmfrt5',
                     'rubbnglgs5', 'wrserest5', 'wrseltr5', 'feelngbstr5', 'tired5', 'mosttired4', 'feelngbstpk5',
                     'types5', 'slpapnea5', 'cpap5', 'dntaldv5', 'uvula5', 'insmnia5', 'rstlesslgs5',
                     'wrksched5', 'extrahrs5'])
        force_num.extend(['sleepage5c', 'wkendsleepdur5t', 'nap5', 'whiirs5c', 'epslpscl5c', 'hoostmeq5c'])

    return force_cat, force_num


def get_testname(experiment_filename):
    stemname = Path(experiment_filename).stem
    print(stemname)

    if stemname.endswith(".pkl"):
        stemname = stemname.replace(".pkl", "_test.csv.gz")

    elif stemname.endswith(".pkl.gz"):
        stemname = stemname.replace(".pkl.gz", "_test.csv.gz")

    elif stemname.endswith(".csv"):
        stemname = stemname.replace(".csv", "_test.csv.gz")

    else:
        stemname += "_test.csv.gz"

    return experiment_filename.replace(os.path.basename(experiment_filename), stemname)


def get_trainname(experiment_filename):
    stemname = Path(experiment_filename).stem
    print(stemname)

    if stemname.endswith(".pkl"):
        stemname = stemname.replace(".pkl", ".csv.gz")

    elif stemname.endswith(".pkl.gz"):
        stemname = stemname.replace(".pkl.gz", ".csv.gz")

    elif stemname.endswith(".csv"):
        stemname = stemname.replace(".csv", ".csv.gz")

    else:
        stemname += ".csv.gz"

    return experiment_filename.replace(os.path.basename(experiment_filename), stemname)


def zip_pkl(filename):

    with open(filename + ".pkl", 'rb') as f_in:
        with gzip.open(filename + ".pkl.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(filename + ".pkl")


def unzip_pkl(filename, remove_zipped=False):
    with gzip.open(filename + ".pkl.gz", 'rb') as f_in:
        with open(filename + ".pkl", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if remove_zipped:
        os.remove(filename + ".pkl.gz")


def delete_pkl(filename, zipped=False):
    if zipped:
        os.remove(filename + ".pkl.gz")
    else:
        os.remove(filename + ".pkl")
# -


