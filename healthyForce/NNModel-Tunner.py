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
from argparse import Namespace
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only, seed

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

from collections import OrderedDict
from datetime import datetime

from MTLModels import MTLRandom, MTLEqual, MTLBandit, MTLDWA, MTLUncertanty
from NNModel import eval_n_times, MyNet, MyTwoStepsNet, get_data, myXYDataset, calculate_classification_metrics

import os

os.environ["SLURM_JOB_NAME"] = "bash"


def get_env_var(varname, default):
    return int(os.environ.get(varname)) if os.environ.get(varname) is not None else default

def chunks(l, n):
    n = len(l) // n
    return [l[i:i+n] for i in range(0, len(l), max(1, n))]

def hyper_tuner(config, NetClass, dataset, ngpus):
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]

    if "structure" in config and config["structure"] == "mpl":
        structure = {"mpl": "mpl"}
    elif "structure" in config and config["structure"] == "lstm":
        structure = {"lstm": {"bidirectional": config["bidirectional"], "hidden_dim": config["hidden_dim"],
                              "num_layers": config["num_layers"]}}
    else:
        structure = {"cnnlstm": {"bidirectional": config["bidirectional"], "hidden_dim": config["hidden_dim"],
                                 "num_layers": config["num_layers"]}}

    monitor = config["monitor"]
    shared_output_size = config["shared_output_size"]
    opt_step_size = config["opt_step_size"]
    weight_decay = config["weight_decay"]
    dropout_input_layers = config["dropout_input_layers"]
    dropout_inner_layers = config["dropout_inner_layers"]

    sleep_metrics = config['sleep_metrics']
    loss_method = 'equal'

    X, Y = get_data(dataset)

    train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True,
                       num_workers=8)
    val = DataLoader(myXYDataset(X["val"], Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True,
                     num_workers=8)

    seed.seed_everything(42)

    path_ckps = "./lightning_logs/test/"

    if monitor == "mcc":
        early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='mcc', mode='max', patience=5)
        ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False,
                              prefix="",
                              monitor="mcc", mode="max")
    else:
        early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min', patience=5)
        ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False,
                              prefix="",
                              monitor="loss", mode="min")

    hparams = Namespace(batch_size=batch_size,
                        shared_output_size=shared_output_size,
                        sleep_metrics=sleep_metrics,
                        dropout_input_layers=dropout_input_layers,
                        dropout_inner_layers=dropout_inner_layers,
                        structure=structure,
                        #
                        # Optmizer configs
                        #
                        opt_learning_rate=learning_rate,
                        opt_weight_decay=weight_decay,
                        opt_step_size=opt_step_size,
                        opt_gamma=0.5,
                        #
                        # Loss combination method
                        #
                        loss_method=loss_method,  # Options: equal, alex, dwa
                        #
                        # Output layer
                        #
                        output_strategy="linear",  # Options: attention, linear
                        dataset=dataset,
                        monitor=monitor,
                        )

    model = NetClass(hparams)
    model.double()

    tune_metrics = {"loss": "loss", "mcc": "mcc", "acc": "acc", "macroF1": "macroF1"}
    tune_cb = TuneReportCallback(tune_metrics, on="validation_end")

    trainer = Trainer(gpus=ngpus, min_epochs=2, max_epochs=100,
                      callbacks=[early_stop_callback, ckp, tune_cb])
    trainer.fit(model, train, val)


def run_tuning_procedure(config, expname, ntrials, ncpus, ngpus, NetClass, dataset="hchs"):
    trainable = tune.with_parameters(hyper_tuner, NetClass=NetClass, dataset=dataset, ngpus=ngpus)

    analysis = tune.run(trainable,
                        resources_per_trial={"cpu": ncpus, "gpu": ngpus},
                        metric="loss",
                        mode="min",
                        config=config,
                        num_samples=ntrials,
                        name=expname)

    print("Best Parameters:", analysis.best_config)

    analysis.results_df["sleep_metrics"] = '_'.join(config["sleep_metrics"])
    analysis.best_result_df["sleep_metrics"] = '_'.join(config["sleep_metrics"])
    analysis.results_df["expname"] = expname
    analysis.best_result_df["expname"] = expname

    analysis.best_result_df.to_csv("best_parameters_exp%s_trials%d.csv" % (expname, ntrials))
    analysis.results_df.to_csv("all_results_exp%s_trials%d.csv" % (expname, ntrials))
    print("Best 5 results")
    print(analysis.results_df.sort_values(by="mcc", ascending=False).head(5))

    return analysis.best_result_df


# +
if __name__ == "__main__":

    SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
    SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
    SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)

    default_mpl = {
        "structure": "mpl",
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "monitor": tune.choice(["loss", "mcc"]),
        "shared_output_size": tune.randint(2, 256),
        "opt_step_size": tune.randint(1, 20),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "dropout_input_layers": tune.uniform(0, 1),
        "dropout_inner_layers": tune.uniform(0, 1),
    }

    default_lstm = {
        "structure": "lstm",
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "bidirectional": tune.choice([True, False]),
        "num_layers": tune.choice([1, 2]),
        "hidden_dim": tune.choice([1024, 512, 256, 128, 64, 32]),
        "monitor": tune.choice(["loss", "mcc"]),
        "shared_output_size": tune.randint(2, 256),
        "opt_step_size": tune.randint(1, 20),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "dropout_input_layers": tune.uniform(0, 1),
        "dropout_inner_layers": tune.uniform(0, 1),
    }

    default_cnnlstm = {
        "structure": "cnnlstm",
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "bidirectional": tune.choice([True, False]),
        "num_layers": tune.choice([1, 2]),
        "hidden_dim": tune.choice([1024, 512, 256, 128, 64, 32]),
        "monitor": tune.choice(["loss", "mcc"]),
        "shared_output_size": tune.randint(2, 256),
        "opt_step_size": tune.randint(1, 20),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "dropout_input_layers": tune.uniform(0, 1),
        "dropout_inner_layers": tune.uniform(0, 1),
    }

    # +
    ncpus = 12
    ngpus = 1
    ntrials = 1000

    # exp_idx = 0 # int(sys.argv[1])
    # sm = "sleepEfficiency" # sys.argv[2]
    # dataset = "mesa" # sys.argv[3]

    experiment_list = [
        [MyNet, default_cnnlstm, "MyNetCNNLSTM"],
        [MyTwoStepsNet, default_cnnlstm, "MyTwoStepsNetCNNLSTM"],
        [MyNet, default_mpl, "MyNetMPL"],
        [MyTwoStepsNet, default_mpl, "MyTwoStepsNetMPL"],
        [MyNet, default_lstm, "MyNetLSTM"],
        [MyTwoStepsNet, default_lstm, "MyTwoStepsNetLSTM"],
    ]

    combinations = []
    for dataset in ["mesa", "hchs"]:
        for sm in ["sleepEfficiency", "awakening", "totalSleepTime", "combined", "all"]:
            for exp_idx in range(6):
                combinations.append([dataset, sm, exp_idx])

    selected_comb = chunks(combinations, SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]

    for comb in selected_comb:
        dataset, sm, exp_idx = comb
        print(type(dataset), type(sm), type(exp_idx))

        NetClass = experiment_list[exp_idx][0]
        config = experiment_list[exp_idx][1]
        exp_name = experiment_list[exp_idx][2]
        exp_name = exp_name + {'combined': "_COM", 'awakening': "_AWE", 'totalSleepTime': "_TST",
                               "sleepEfficiency": "_EFF", "all": "_ALL"}[sm]
        exp_name = exp_name + "_" + dataset

        config["sleep_metrics"] = {'combined': ['combined'], 'sleepEfficiency': ['sleepEfficiency'],
                                   'awakening': ['awakening'], 'totalSleepTime': ['totalSleepTime'],
                                   'all': ['sleepEfficiency', 'awakening', 'totalSleepTime', 'combined']}[sm]

        best_df = run_tuning_procedure(config, exp_name, ntrials=ntrials, ncpus=ncpus,
                                       ngpus=ngpus, NetClass=NetClass, dataset=dataset)

        keys = [k for k in best_df.keys() if "config." in k]
        best_parameters = {}
        for k in keys:
            best_parameters[k.split("config.")[1]] = best_df[k].iloc[0]

        print("Final evaluation:")
        results_MyNet_MP = eval_n_times(best_parameters, NetClass, n=10, gpus=0, patience=10)
        results_MyNet_MP["sleep_metrics"] = sm
        results_MyNet_MP["expname"] = exp_name
        results_MyNet_MP["dataset"] = dataset
        results_MyNet_MP["ntrials"] = ntrials

        print(results_MyNet_MP)
        results_MyNet_MP.to_csv("final_%s.csv" % exp_name)

#eval_n_times(config, NetClass, 1, gpus=0, patience=1)

# -

# # Experiments with best parameters

# best_MyNet_MPL = {'structure': 'mpl', 'learning_rate': 0.0007051729202516784, 'batch_size': 32, 'monitor': 'loss', 'shared_output_size': 197, 'opt_step_size': 19, 'weight_decay': 0.00023639567148799848, 'dropout_input_layers': 0.26128234774061, 'dropout_inner_layers': 0.5797312209649872}
# results_MyNet_MP = eval_n_times(best_MyNet_MPL, MyNet, 30, gpus=1, patience=10)
#
# results_MyNet_MP
#
# best_MyTwoStepsNet_MPL = {'structure': 'mpl', 'learning_rate': 0.0007051729202516784, 'batch_size': 32, 'monitor': 'loss', 'shared_output_size': 197, 'opt_step_size': 19, 'weight_decay': 0.00023639567148799848, 'dropout_input_layers': 0.26128234774061, 'dropout_inner_layers': 0.5797312209649872}
# results_MyTwoStepsNet_MP = eval_n_times(best_MyTwoStepsNet_MPL, MyTwoStepsNet, 10, gpus=1, patience=10)
#
# results_MyTwoStepsNet_MP
#
# # +
# # EXP 3: (processing)
# best_MyNet_LSTM_itMinus1 = {'structure': 'lstm', 'learning_rate': 0.0011414609495356569, 'batch_size': 128, 'bidirectional': True, 'num_layers': 2, 'hidden_dim': 512, 'monitor': 'loss', 'shared_output_size': 223, 'opt_step_size': 2, 'weight_decay': 7.32627848475396e-05, 'dropout_input_layers': 0.9168838231595683, 'dropout_inner_layers': 0.5900610092330482}
#
#
# best_MyNet_LSTM_it1 = {'structure': 'lstm', 'learning_rate': 0.0003192446799812686, 'batch_size': 128, 'bidirectional': False, 'num_layers': 2, 'hidden_dim': 1024, 'monitor': 'loss', 'shared_output_size': 245, 'opt_step_size': 12, 'weight_decay': 0.0007532163355802958, 'dropout_input_layers': 0.8549525509755683, 'dropout_inner_layers': 0.43791565543235333}
# best_MyNet_LSTM_it2 = {'structure': 'lstm', 'learning_rate': 0.0029602558993806825, 'batch_size': 128, 'bidirectional': False, 'num_layers': 2, 'hidden_dim': 1024, 'monitor': 'mcc', 'shared_output_size': 90, 'opt_step_size': 16, 'weight_decay': 8.805867549384574e-05, 'dropout_input_layers': 0.7994944074580097, 'dropout_inner_layers': 0.46902262176752973}
# best_MyNet_LSTM_it3 = {'structure': 'lstm', 'learning_rate': 0.0006019951586551069, 'batch_size': 128, 'bidirectional': False, 'num_layers': 2, 'hidden_dim': 1024, 'monitor': 'loss', 'shared_output_size': 81, 'opt_step_size': 11, 'weight_decay': 1.4861273916829396e-05, 'dropout_input_layers': 0.898288853089049, 'dropout_inner_layers': 0.5874479212693492}
# best_MyNet_LSTM_it4 = {'structure': 'lstm', 'learning_rate': 0.0008776481997236678, 'batch_size': 32, 'bidirectional': False, 'num_layers': 2, 'hidden_dim': 256, 'monitor': 'mcc', 'shared_output_size': 251, 'opt_step_size': 12, 'weight_decay': 0.001464800495673266, 'dropout_input_layers': 0.8390916287947863, 'dropout_inner_layers': 0.5694863291249453}
#
#
# results_MyNet_LSTM = eval_n_times(best_MyNet_LSTM_it4, MyNet, 10, gpus=0, patience=10)




# best_MyNet_LSTM_itMinus1
# acc_sleepEfficiency        0.807924
# macrof1_sleepEfficiency    0.654318
# mcc_sleepEfficiency        0.557096
# average_global             0.557096
# test_loss                  0.450955

# best_MyNet_LSTM_it4 (partial)
# acc_sleepEfficiency        0.820982
# macrof1_sleepEfficiency    0.720677
# mcc_sleepEfficiency        0.593537
# average_global             0.593537
# test_loss                  0.441487

# best_MyNet_LSTM_it3
# acc_sleepEfficiency        0.821987
# macrof1_sleepEfficiency    0.716940
# mcc_sleepEfficiency        0.593036
# average_global             0.593036
# test_loss                  0.427165

# best_MyNet_LSTM_it2
# acc_sleepEfficiency        0.760045
# macrof1_sleepEfficiency    0.653846
# mcc_sleepEfficiency        0.525337
# average_global             0.525337
# test_loss                  0.532730

# best_MyNet_LSTM_it1
# acc_sleepEfficiency        0.812165
# macrof1_sleepEfficiency    0.644434
# mcc_sleepEfficiency        0.568586
# average_global             0.568586
# test_loss                  0.452630

# -

# EXP 4:
# best_MyTwoStepsNet_LSTM = {'structure': 'lstm', 'learning_rate': 3.46625048422162e-05, 'batch_size': 32, 'bidirectional': True, 'num_layers': 1, 'hidden_dim': 256, 'monitor': 'loss', 'shared_output_size': 35, 'opt_step_size': 7, 'weight_decay': 1.2369444260214379e-05, 'dropout_input_layers': 0.015215555678978587, 'dropout_inner_layers': 0.19178887239193476}
# results_MyTwoStepsNet_LSTM = eval_n_times(best_MyTwoStepsNet_LSTM, MyTwoStepsNet, 10, gpus=1, patience=10)

# +
# config_lstm = {'learning_rate': 0.001532635835186596, 'batch_size': 128, 'bidirectional': True, 'num_layers': 2, 'monitor': 'mcc', 'shared_output_size': 64, 'opt_step_size': 5
# , 'weight_decay': 0.0002774925005331883, 'dropout_input_layers': 0.9030694535320151, 'dropout_inner_layers': 0.6805005401772247, 'hidden_dim': 128}
# results2 = eval_n_times(config_lstm, MyNet, 10, gpus=1)

# best_parameters_20k_lstm = {'learning_rate': 0.001532635835186596, 'batch_size': 128, 'bidirectional': True, 'num_layers': 2, 'monitor': 'mcc', 'shared_output_size': 64, 'opt_step_size': 5, 'weight_decay': 0.0002774925005331883, 'dropout_input_layers': 0.9030694535320151, 'dropout_inner_layers': 0.6805005401772247, 'hidden_dim': 128}
# results = eval_n_times(best_parameters_20k_lstm, MyNet, 10, gpus=1, patience=10)



# +
# config = {'learning_rate': 0.002033717318575583, 'batch_size': 128, 'bidirectional': False, 'num_layers': 2, 'monitor': 'mcc', 'shared_output_size': 8, 'opt_step_size': 10, 'weight_decay': 0.00044602388560120504, 'dropout_input_layers': 0.7777008089936246, 'dropout_inner_layers': 0.4858974282110715, 'hidden_dim': 1024}
# eval_n_times(10, config)

# +
# config = {'learning_rate': 0.002, 'batch_size': 128, 'bidirectional': False, 'num_layers': 2, 
#           'monitor': 'mcc', 'shared_output_size': 8, 'opt_step_size': 10, 
#           'weight_decay': 0.0005, 'dropout_input_layers': 0.75, 'dropout_inner_layers': 0.5, 
#           'hidden_dim': 1024}
# eval_n_times(10, config)

# +
#config = {'learning_rate': 0.0019678250804352564, 'batch_size': 64, 'bidirectional': False, 'num_layers': 2, 'monitor': 'mcc', 'shared_output_size': 128, 'opt_step_size': 
#10, 'weight_decay': 0.00014710600472436756, 'dropout_input_layers': 0.6296084429082514, 'dropout_inner_layers': 0.15221034364037567, 'hidden_dim': 512}
#results = eval_n_times(10, config)
# config = {'sleep_metrics': ["sleepEfficiency", "awakening", "totalSleepTime", "combined"],
#     'learning_rate': 0.002, 'batch_size': 64, 'bidirectional': False, 'num_layers': 2, 'monitor': 'mcc', 'shared_output_size': 128, 'opt_step_size': 10, 'weight_decay': 0.00015, 'dropout_input_layers': 0.63, 'dropout_inner_layers': 0.15, 'hidden_dim': 512}
# results3 = eval_n_times(10, config)

# +
# config = {'structure': 'mpl', 'learning_rate': 8.573090880810072e-06, 'batch_size': 64, 'monitor': 'loss', 'shared_output_size': 229, 'opt_step_size': 3, 'weight_decay': 0.00040127864779085804, 'dropout_input_layers': 0.030797573274495726, 'dropout_inner_layers': 0.35920674647401696}
# results3 = eval_n_times(10, config)
# results3 

# +
# config = {
#     "learning_rate": tune.loguniform(1e-4, 1e-1),
#     "batch_size": tune.choice([32, 64, 128]),
#     "bidirectional": tune.choice([True, False]),
#     "num_layers": tune.choice([1, 2]),
#     "monitor": tune.choice(["loss", "mcc"]),
#     "shared_output_size": tune.choice([8, 16, 32, 64, 128]),
#     "opt_step_size": tune.choice([5, 10]),
#     "weight_decay": tune.loguniform(1e-4, 1e-2),
#     "dropout_input_layers": tune.uniform(0, 1),
#     "dropout_inner_layers": tune.uniform(0, 1),
#     "hidden_dim": tune.choice([1024, 512, 256, 128, 64, 32]),
# }


# trainable = tune.with_parameters(hyper_tuner, dataset="hchs")

# analysis = tune.run(trainable,
#                     resources_per_trial={"cpu": 8, "gpu": 1},
#                     metric="loss",
#                     mode="min",
#                     config=config,
#                     num_samples=3,
#                     name="tune_hchs")

# print("Best Parameters:", analysis.best_config)

# analysis.best_result_df.to_csv("best_parameters_%s.csv" % analysis.best_trial)
