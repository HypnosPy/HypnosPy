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
from pytorch_lightning.utilities import rank_zero_only, seed

from collections import OrderedDict
from datetime import datetime

from MTLModels import MTLRandom, MTLEqual, MTLBandit, MTLDWA, MTLUncertanty


# -
def get_data(DATASET="hchs"):
    # fullpath = "/home/palotti/github/HypnosPy/healthyForce/"
    fullpath = "/export/sc2/jpalotti/github/HypnosPy/healthyForce/"
    with open(fullpath + "%s_to_NN.pkl" % (DATASET), "rb") as f:
        X, Y = pickle.load(f)
    return X, Y


def eval_n_times(config, NetClass, n, gpus=1, patience=3, dataset = "hchs"):

    learning_rate = float(config["learning_rate"])
    batch_size = int(config["batch_size"])

    if "structure" in config and config["structure"] == "mpl":
        structure = {"mpl": "mpl"}
    else:
        structure = {"lstm": {"bidirectional": bool(config["bidirectional"]), "hidden_dim": int(config["hidden_dim"]),
                              "num_layers": int(config["num_layers"])}}
    monitor = config["monitor"]
    shared_output_size = int(config["shared_output_size"])
    opt_step_size = int(config["opt_step_size"])
    weight_decay = float(config["weight_decay"])
    dropout_input_layers = float(config["dropout_input_layers"])
    dropout_inner_layers = float(config["dropout_inner_layers"])

    if "sleep_metrics" in config:
        sleep_metrics = config['sleep_metrics']
    else:
        sleep_metrics = ['sleepEfficiency']
    loss_method = 'equal'

    X, Y = get_data(dataset)
    train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True,
                       num_workers=8)
    val = DataLoader(myXYDataset(X["val"], Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True,
                     num_workers=8)
    test = DataLoader(myXYDataset(X["test"], Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True,
                      num_workers=8)

    results = []
    for s in range(n):
        seed.seed_everything(s)

        path_ckps = "./lightning_logs/test/"

        if monitor == "mcc":
            early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='mcc', mode='max',
                                                patience=patience)
            ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False,
                                  prefix="",
                                  monitor="mcc", mode="max")
        else:
            early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min',
                                                patience=patience)
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

        trainer = Trainer(gpus=gpus, min_epochs=2,
                          max_epochs=300,
                          callbacks=[early_stop_callback, ckp])
        trainer.fit(model, train, val)
        res = trainer.test(test_dataloaders=test)
        results.append(res[0])

    return pd.DataFrame(results)


class myXYDataset(Dataset):
    def __init__(self, X, Y):
        self.Y = Y
        self.X = X

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x, y = {}, {}
        for feature_set in self.X.keys():
            x[feature_set] = self.X[feature_set].iloc[idx].values.astype(np.double)

        for sleep_metric in self.Y.keys():
            y[sleep_metric] = self.Y[sleep_metric].iloc[idx]

        return x, y


def calculate_classification_metrics(labels, predictions):
    return metrics.accuracy_score(labels, predictions), \
           metrics.f1_score(labels, predictions, average='macro', labels=[0, 1, 2]), \
           metrics.f1_score(labels, predictions, average='micro', labels=[0, 1, 2]), \
           metrics.matthews_corrcoef(labels, predictions)


# +
class NetDemographics(pl.LightningModule):

    def __init__(self,
                 input_size=201,
                 output_dim=4,
                 dropout_rate=0.0,
                 ):
        super(NetDemographics, self).__init__()

        self.lin1 = nn.Linear(input_size, 128)
        self.lin2 = nn.Linear(128, 32)
        self.lin3 = nn.Linear(32, output_dim)
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.drop(x)
        x = self.act(self.lin1(x))
        x = self.drop(x)
        x = self.act(self.lin2(x))
        x = self.drop(x)
        x = self.act(self.lin3(x))
        return x


class Net4Input(pl.LightningModule):
    # Used for 'ae24', 'vae24'
    def __init__(self,
                 input_size=4,
                 output_dim=2,
                 dropout_rate=0.0,
                 ):
        super(Net4Input, self).__init__()

        self.lin1 = nn.Linear(input_size, 2)
        self.lin2 = nn.Linear(2, output_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.drop(x)
        x = self.act(self.lin1(x))
        x = self.drop(x)
        x = self.act(self.lin2(x))
        return x

class Net8Input(pl.LightningModule):

    def __init__(self,
                 input_size=8,
                 output_dim=2,
                 dropout_rate=0.0,
                 ):
        super(Net8Input, self).__init__()

        self.lin1 = nn.Linear(input_size, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, output_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.drop(x)
        x = self.act(self.lin1(x))
        x = self.drop(x)
        x = self.act(self.lin2(x))
        x = self.drop(x)
        x = self.act(self.lin3(x))
        return x

class Net16Input(pl.LightningModule):
    #
    def __init__(self,
                 input_size=16,
                 output_dim=2,
                 dropout_rate=0.0,
                 ):
        super(Net16Input, self).__init__()

        self.lin1 = nn.Linear(input_size, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, output_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.drop(x)
        x = self.act(self.lin1(x))
        x = self.drop(x)
        x = self.act(self.lin2(x))
        x = self.drop(x)
        x = self.act(self.lin3(x))

        return x

class Net64Input(pl.LightningModule):
    def __init__(self,
                 input_size=64,
                 output_dim=2,
                 dropout_rate=0.0,
                 ):
        super(Net64Input, self).__init__()

        self.lin1 = nn.Linear(input_size, 32)
        self.lin2 = nn.Linear(32, 8)
        self.lin3 = nn.Linear(8, output_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.drop(x)
        x = self.act(self.lin1(x))
        x = self.drop(x)
        x = self.act(self.lin2(x))
        x = self.drop(x)
        x = self.act(self.lin3(x))
        return x


class Net128Input(pl.LightningModule):
    def __init__(self,
                 input_size=128,
                 output_dim=2,
                 dropout_rate=0.0,
                 ):
        super(Net128Input, self).__init__()

        self.lin1 = nn.Linear(input_size, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, output_dim)  # 8)
        # self.lin4 = nn.Linear(8, output_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.drop(x)
        x = self.act(self.lin1(x))
        x = self.drop(x)
        x = self.act(self.lin2(x))
        x = self.drop(x)
        x = self.act(self.lin3(x))
        return x


def get_number_internal_layers(n, output_size):
    """
    E.g.:
        get_number_internal_layers(20, 3) --> [16, 8, 4]
        get_number_internal_layers(192, 16) # --> [128, 64, 32]
    """
    i = 1
    d = 2
    s = []
    while (n - 1) / d > 1:
        s.append(d)
        i += 1
        d = 2 ** i

    s = [e for e in s if e > output_size]
    return s[::-1]


class CNNLayer(pl.LightningModule):
    def __init__(self,
                 kernel_size=11,
                 ):
        super(CNNLayer, self).__init__()

        self.cnn = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding="same")
        self.batch = nn.BatchNorm1d(1)
        self.relu = nn.ReLU(inplace=True)

        # nn.Sequential()

    def forward(self, x):
        x = x.unsqueeze(1)  # Reshape from (B,L) to (B, C=1, L)
        x = self.cnn(x)
        x = self.batch(x)
        x = self.relu(x)
        x = x.squeeze(1)  # Reshape it back to (B,L)
        return x


class LSTMLayer(pl.LightningModule):
    def __init__(self,
                 input_size=8,
                 hidden_dim=10,
                 output_dim=2,
                 dropout_rate=0.0,
                 break_point=None,
                 bidirectional=False,
                 num_layers=1,
                 ):
        super(LSTMLayer, self).__init__()

        if break_point is None:
            break_point = input_size

        print("BREAK POINT:", break_point)

        if num_layers == 1:
            dropout_rate = 0.0

        self.lstm = nn.LSTM(break_point, hidden_dim, num_layers=num_layers, dropout=dropout_rate,
                            batch_first=True, bidirectional=bidirectional)
        self.linlayers = nn.ModuleList()

        if bidirectional:
            hidden_dim *= 2

        last_d = hidden_dim * (input_size // break_point)
        for lay_size in get_number_internal_layers(last_d, output_dim):
            print("Last: %d, Next: %d" % (last_d, lay_size))
            self.linlayers.append(nn.Linear(last_d, lay_size))
            last_d = lay_size

        print("Very Last: %d, Out: %d" % (last_d, output_dim))
        print("#Lin layers: ", len(self.linlayers))
        self.last_lin = nn.Linear(last_d, output_dim)
        self.act = nn.ReLU()
        self.break_point = break_point

    def forward(self, x):
        # print("INPUT:", x.shape)
        x = x.view(x.shape[0], x.shape[1] // self.break_point, -1)
        # print("Reshaped to:", x.shape)

        x, hidden = self.lstm(x)
        # print("After LSTM:", x.shape)

        # x = x.squeeze()
        x = x.reshape(x.shape[0], -1)

        # print("After reshape:", x.shape)
        for lay in self.linlayers:
            x = self.act(lay(x))

        x = self.act(self.last_lin(x))
        return x

# +
class MyNet(pl.LightningModule):

    def __init__(self, hparams):

        super().__init__()

        self.save_hyperparameters()
        saved_results = None

        self.hparams = hparams
        self.timestamp = datetime.now()

        # Optimizer configs
        self.opt_learning_rate = hparams.opt_learning_rate
        self.opt_weight_decay = hparams.opt_weight_decay
        self.opt_step_size = hparams.opt_step_size
        self.opt_gamma = hparams.opt_gamma
        self.dropout_input_layers = hparams.dropout_input_layers
        self.dropout_inner_layers = hparams.dropout_inner_layers

        # Other configs
        self.dataset = hparams.dataset
        self.batch_size = hparams.batch_size
        self.shared_output_size = hparams.shared_output_size
        self.sleep_metrics = hparams.sleep_metrics
        self.loss_method = hparams.loss_method
        self.structure = hparams.structure

        self.feature_sets = ["bouts", "demo", "bins", "cosinor",
                             "stats", "time", "ae24", "vae24",
                             "hourly_bins", "hourly_stats", "hourly_bouts"
                             ]

        self.input_size = 1232 if self.dataset == "hchs" else 908

        if list(self.structure.keys())[0] == "lstm":
            self.net = LSTMLayer(input_size=self.input_size, break_point=self.input_size,
                                 dropout_rate=self.dropout_input_layers,
                                 hidden_dim=self.structure["lstm"]["hidden_dim"],
                                 bidirectional=self.structure["lstm"]["bidirectional"],
                                 num_layers=self.structure["lstm"]["num_layers"],
                                 output_dim=256,
                                 )

        elif list(self.structure.keys())[0] == "cnnlstm":
            self.net = nn.Sequential(
                CNNLayer(kernel_size=11),
                LSTMLayer(input_size=self.input_size, break_point=self.input_size,
                          dropout_rate=self.dropout_input_layers,
                          hidden_dim=self.structure["cnnlstm"]["hidden_dim"],
                          bidirectional=self.structure["cnnlstm"]["bidirectional"],
                          num_layers=self.structure["cnnlstm"]["num_layers"],
                          output_dim=256,
                          )
            )

        else:
            self.net = nn.Sequential(OrderedDict([
                ('lin1', nn.Linear(self.input_size, 512)),
                ('act1', nn.ReLU(inplace=True)),
                ('dropout', nn.Dropout(self.dropout_inner_layers)),
                ('lin2', nn.Linear(512, 256)),
                ('act2', nn.ReLU(inplace=True)),
                ('dropout', nn.Dropout(self.dropout_inner_layers)),
            ]))

        self.drop = nn.Dropout(self.dropout_inner_layers)
        self.shared = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(256, 64)),
            ('act1', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(self.dropout_inner_layers)),
            ('lin2', nn.Linear(64, self.shared_output_size)),
            ('act2', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(self.dropout_inner_layers)),
        ]))

        self.output = nn.ModuleDict()
        for metric in self.sleep_metrics:
            if hparams.output_strategy == "linear":
                self.output[metric] = nn.Sequential(nn.Linear(self.shared_output_size, 16),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(16, 8),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(8, 3)
                                                    )

        # Logsigma loss from (Alex kendall's paper)
        print("Using loss aggregator: %s" % (self.loss_method))

        if self.loss_method == "equal" or self.loss_method is None:
            self.loss_aggregator = MTLEqual(len(self.sleep_metrics))

        elif self.loss_method == "alex":
            self.loss_aggregator = MTLUncertanty(len(self.sleep_metrics))
            # self.loss_aggregator.logsigma = self.loss_aggregator.logsigma.to("cuda:0")

            self.mylogsigma = nn.Parameter(torch.zeros(len(self.sleep_metrics)))
            # Link parameters to MTLUncertanty model:
            self.loss_aggregator.logsigma = self.mylogsigma
            # self.logsigma = nn.Parameter(torch.zeros(len(self.sleep_metrics)))

        elif self.loss_method.startswith("dwa"):
            self.loss_aggregator = MTLDWA(len(self.sleep_metrics), self.loss_method.split("_")[1])

        elif "bandit" in self.loss_method:
            self.loss_aggregator = MTLBandit(len(self.sleep_metrics),
                                             strategy=self.loss_method, )

        # Save the results every epoch
        self.saved_results = []

    def forward(self, x_):
        x = torch.cat((x_["bouts"], x_["demo"], x_["bins"], x_["cosinor"], x_["stats"], x_["time"],
                       x_["ae24"], x_["vae24"], x_["hourly_bins"], x_["hourly_stats"], x_["hourly_bouts"]), 1)
        # print("SHAPE:", x.shape)
        x = self.net(x)
        x = self.drop(x)
        x = self.shared(x)
        x = x.view(x.size(0), -1)

        out = {}
        for metric in self.sleep_metrics:
            out[metric] = self.output[metric](x)

        return out

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3, eps=1e-08)
        print("Current number of parameters:", len(list(self.parameters())))
        optimizer = optim.Adam(self.parameters(), lr=self.opt_learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         patience=self.opt_step_size,
                                                         factor=self.opt_gamma,  # new_lr = lr * factor (default = 0.1)
                                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'loss'}

    def calculate_losses(self, y, predictions):
        loss_fnct = nn.CrossEntropyLoss()

        losses = {}
        losses_list = []

        for metric in self.sleep_metrics:
            loss = loss_fnct(predictions[metric], y[metric])
            losses[metric] = loss
            losses_list.append(loss)

        total_loss = self.loss_aggregator.aggregate_losses(losses_list)
        total_loss = total_loss.type_as(loss)

        return losses, total_loss

    def process_step(self, batch):
        x, y = batch
        x_ = {}
        for fset in self.feature_sets:
            x_[fset] = x[fset].view(x[fset].size(0), -1)

        predictions = self(x_)
        losses, total_loss = self.calculate_losses(y, predictions)
        return predictions, losses, total_loss, y

    def training_step(self, batch, batch_idx):
        predictions, losses, total_loss, y = self.process_step(batch)
        # self.log('loss', total_loss)
        return {'loss': total_loss, 'ys': y, 'preds': predictions, }

    def validation_step(self, batch, batch_idx):
        predictions, losses, total_loss, y = self.process_step(batch)
        # self.log('val_loss', total_loss)
        return {'loss': total_loss, 'ys': y, 'preds': predictions, 'loss_per_metric': losses}

    def test_step(self, batch, batch_idx):
        predictions, losses, total_loss, y = self.process_step(batch)
        return {'loss': total_loss, 'ys': y, 'preds': predictions, 'loss_per_metric': losses}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([row['loss'] for row in outputs]).mean()
        print("(Validation) Total Loss: %.4f" % val_loss)

        avg_metrics = {"acc": [], "macroF1": [], "mcc": [], "loss": []}
        avg_losses = []
        for metric in self.sleep_metrics:
            y = torch.stack([row["ys"][metric] for row in outputs]).view(-1).cpu()
            pred = torch.stack([row['preds'][metric].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()

            avg_loss = torch.stack([row['loss_per_metric'][metric] for row in outputs]).mean()

            acc, macrof1, microf1, mcc = calculate_classification_metrics(y, pred)
            print("(%s) Epoch: %d, ACC: %.3f, MacroF1: %.3f, MCC: %.3f, Loss: %.4f" % (
                metric, self.current_epoch, acc, macrof1, mcc, avg_loss))

            avg_metrics["acc"].append(acc)
            avg_metrics["macroF1"].append(macrof1)
            avg_metrics["mcc"].append(mcc)
            avg_metrics["loss"].append(avg_loss.cpu().numpy())

        if self.loss_method is not None:
            if self.loss_method == "alex":
                print("LOGSIGMAS:", self.loss_aggregator.logsigma)

            elif self.loss_method.startswith("dwa") or self.loss_method.startswith("bandit"):
                self.loss_aggregator.lambda_weight = self.loss_aggregator.lambda_weight.type_as(avg_losses[0])

        self.loss_aggregator.adjust_after_validation(avg_losses, self.current_epoch)

        for eval_metric in ["acc", "macroF1", "mcc", "loss"]:
            avg_metrics[eval_metric] = np.array(avg_metrics[eval_metric]).mean()
            print("Average %s: %.4f" % (eval_metric, avg_metrics[eval_metric]))
            self.log(eval_metric, avg_metrics[eval_metric])

    def test_epoch_end(self, outputs):
        return_dict = {}
        test_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print("(Test) Total Loss: %.4f" % test_loss)

        metric_global = "mcc"
        opt_metric = []
        for metric in self.sleep_metrics:
            y = torch.stack([row["ys"][metric] for row in outputs]).view(-1).cpu()
            pred = torch.stack([row['preds'][metric].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()
            # pred = torch.stack([row['preds'].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()

            avg_loss = torch.stack([row['loss_per_metric'][metric] for row in outputs]).mean()

            acc, macrof1, microf1, mcc = calculate_classification_metrics(y, pred)
            print("(%s) Epoch: %d, ACC: %.3f, MacroF1: %.3f, MCC: %.3f, Loss: %.4f" %
                  (metric, self.current_epoch, acc, macrof1, mcc, avg_loss))

            return_dict["acc_%s" % metric] = acc
            return_dict["macrof1_%s" % metric] = macrof1
            return_dict["mcc_%s" % metric] = mcc
            opt_metric.append({"macrof1": macrof1, "mcc": mcc}[metric_global])

        average_metric = np.array(opt_metric).mean()
        print("Test Average Eval Metric: %.4f" % average_metric)
        return_dict["average_global"] = average_metric
        return_dict["test_loss"] = test_loss

        row = {}
        row["test"] = True
        for p in ["dataset", "loss_method", "opt_gamma", "opt_learning_rate", "sleep_metrics",
                  "dropout_input_layers", "dropout_inner_layers", "opt_step_size", "opt_weight_decay",
                  "output_strategy", "shared_output_size", "batch_size"]:
            row[p] = self.hparams[p]
        for k, v in return_dict.items():
            row[k] = v

        pd.DataFrame([row]).to_csv("nn_results/nn_test_%s.csv.gz" % (self.timestamp), index=False)

        for k, v in return_dict.items():
            self.log(k, v)


# +
def get_input_size(dataset, fset):
    return {"mesa":
                {"bins": 4, "hourly_bins": 96, "stats": 8, "hourly_stats": 192, "bouts": 16,
                 "hourly_bouts": 372, "time": 8, "cosinor": 3, "demo": 201, "ae_bouts": 8, "ae24": 4,
                 "vae24": 4,
                 },
            "hchs":
                {"bins": 4, "hourly_bins": 96, "stats": 8, "hourly_stats": 192, "bouts": 16,
                 "hourly_bouts": 380, "time": 8, "cosinor": 3, "demo": 517, "ae_bouts": 8, "ae24": 4,
                 "vae24": 4
                 }
            }[dataset][fset]


class MyTwoStepsNet(pl.LightningModule):

    def __init__(self,
                 hparams,
                 ):

        super().__init__()

        self.save_hyperparameters()
        self.hparams = hparams
        self.timestamp = datetime.now()

        # Optimizer configs
        self.opt_learning_rate = hparams.opt_learning_rate
        self.opt_weight_decay = hparams.opt_weight_decay
        self.opt_step_size = hparams.opt_step_size
        self.opt_gamma = hparams.opt_gamma
        self.dropout_input_layers = hparams.dropout_input_layers
        self.dropout_inner_layers = hparams.dropout_inner_layers
        self.structure = hparams.structure

        # Other configs
        self.dataset = hparams.dataset
        self.batch_size = hparams.batch_size
        self.shared_output_size = hparams.shared_output_size
        self.sleep_metrics = hparams.sleep_metrics
        self.loss_method = hparams.loss_method

        self.feature_sets = ["bouts", "demo", "bins", "cosinor",
                             "stats", "time", "ae24", "vae24",
                             "hourly_bins", "hourly_stats", "hourly_bouts"
                             ]

        output_sizes = {"bouts": 2, "demo": 16, "bins": 2, "cosinor": 2, "stats": 4, "time": 2,
                        "ae24": 2, "vae24": 2, "hourly_bins": 8, "hourly_stats": 16, "hourly_bouts": 16}

        self.demographics = NetDemographics(input_size=get_input_size(self.dataset, "demo"),
                                            output_dim=output_sizes["demo"], dropout_rate=self.dropout_input_layers)
        self.bouts = Net16Input(input_size=get_input_size(self.dataset, "bouts"),
                                output_dim=output_sizes["bouts"], dropout_rate=self.dropout_input_layers)
        self.stats = Net8Input(input_size=get_input_size(self.dataset, "stats"),
                               output_dim=output_sizes["stats"], dropout_rate=self.dropout_input_layers)
        self.time = Net8Input(input_size=get_input_size(self.dataset, "time"),
                              output_dim=output_sizes["time"], dropout_rate=self.dropout_input_layers)
        self.ae24 = Net4Input(input_size=get_input_size(self.dataset, "ae24"),
                              output_dim=output_sizes["ae24"], dropout_rate=self.dropout_input_layers)
        self.vae24 = Net4Input(input_size=get_input_size(self.dataset, "vae24"),
                               output_dim=output_sizes["vae24"], dropout_rate=self.dropout_input_layers)
        self.cosinor = Net4Input(input_size=get_input_size(self.dataset, "cosinor"),
                                 output_dim=output_sizes["cosinor"], dropout_rate=self.dropout_input_layers)
        self.bins = Net4Input(input_size=get_input_size(self.dataset, "bins"),
                              output_dim=output_sizes["bins"], dropout_rate=self.dropout_input_layers)

        if list(self.structure.keys())[0] == "lstm":
            self.hourly_bins = LSTMLayer(input_size=get_input_size(self.dataset, "hourly_bins"),
                                         hidden_dim=get_input_size(self.dataset, "hourly_bins") // 2,
                                         output_dim=output_sizes["hourly_bins"], dropout_rate=self.dropout_input_layers)
            self.hourly_stats = LSTMLayer(input_size=get_input_size(self.dataset, "hourly_stats"),
                                          hidden_dim=get_input_size(self.dataset, "hourly_stats") // 2,
                                          output_dim=output_sizes["hourly_stats"],
                                          dropout_rate=self.dropout_input_layers)
            self.hourly_bouts = LSTMLayer(input_size=get_input_size(self.dataset, "hourly_bouts"),
                                          hidden_dim=get_input_size(self.dataset, "hourly_bouts") // 2,
                                          output_dim=output_sizes["hourly_bouts"],
                                          dropout_rate=self.dropout_input_layers,
                                          break_point=get_input_size(self.dataset, "hourly_bouts"))

        elif list(self.structure.keys())[0] == "cnnlstm":

            self.hourly_bins = nn.Sequential(
                CNNLayer(kernel_size=5),
                LSTMLayer(input_size=get_input_size(self.dataset, "hourly_bins"),
                          hidden_dim=get_input_size(self.dataset, "hourly_bins") // 2,
                          output_dim=output_sizes["hourly_bins"], dropout_rate=self.dropout_input_layers)
            )
            self.hourly_stats = nn.Sequential(
                CNNLayer(kernel_size=11),
                LSTMLayer(input_size=get_input_size(self.dataset, "hourly_stats"),
                          hidden_dim=get_input_size(self.dataset, "hourly_stats") // 2,
                          output_dim=output_sizes["hourly_stats"], dropout_rate=self.dropout_input_layers)
            )
            self.hourly_bouts = nn.Sequential(
                CNNLayer(kernel_size=11),
                LSTMLayer(input_size=get_input_size(self.dataset, "hourly_bouts"),
                          hidden_dim=get_input_size(self.dataset, "hourly_bouts") // 2,
                          output_dim=output_sizes["hourly_bouts"], dropout_rate=self.dropout_input_layers,
                          break_point=get_input_size(self.dataset, "hourly_bouts")))
        else:
            self.hourly_bins = Net64Input(input_size=get_input_size(self.dataset, "hourly_bins"),
                                          output_dim=output_sizes["hourly_bins"],
                                          dropout_rate=self.dropout_input_layers)
            self.hourly_stats = Net128Input(input_size=get_input_size(self.dataset, "hourly_stats"),
                                            output_dim=output_sizes["hourly_stats"],
                                            dropout_rate=self.dropout_input_layers)
            self.hourly_bouts = Net128Input(input_size=get_input_size(self.dataset, "hourly_bouts"),
                                            output_dim=output_sizes["hourly_bouts"],
                                            dropout_rate=self.dropout_input_layers)

        print("Internal Layers: ", sum(output_sizes.values()))
        self.drop = nn.Dropout(self.dropout_inner_layers)
        self.shared = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(sum(output_sizes.values()), 64)),
            ('act1', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(self.dropout_inner_layers)),
            ('lin2', nn.Linear(64, self.shared_output_size)),
            ('act2', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(self.dropout_inner_layers)),
        ]))

        self.output = nn.ModuleDict()
        for metric in self.sleep_metrics:
            if hparams.output_strategy == "linear":
                self.output[metric] = nn.Sequential(nn.Linear(self.shared_output_size, 16),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(16, 8),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(8, 3)
                                                    )

        # Logsigma loss from (Alex kendall's paper)
        print("Using loss aggregator: %s" % (self.loss_method))

        if self.loss_method == "equal" or self.loss_method is None:
            self.loss_aggregator = MTLEqual(len(self.sleep_metrics))

        elif self.loss_method == "alex":
            self.loss_aggregator = MTLUncertanty(len(self.sleep_metrics))
            # self.loss_aggregator.logsigma = self.loss_aggregator.logsigma.to("cuda:0")

            self.mylogsigma = nn.Parameter(torch.zeros(len(self.sleep_metrics)))
            # Link parameters to MTLUncertanty model:
            self.loss_aggregator.logsigma = self.mylogsigma
            # self.logsigma = nn.Parameter(torch.zeros(len(self.sleep_metrics)))

        elif self.loss_method.startswith("dwa"):
            self.loss_aggregator = MTLDWA(len(self.sleep_metrics), self.loss_method.split("_")[1])

        elif "bandit" in self.loss_method:
            self.loss_aggregator = MTLBandit(len(self.sleep_metrics),
                                             strategy=self.loss_method, )

        # Save the results every epoch
        self.saved_results = []

    def forward(self, x_):
        x = torch.cat((self.bouts(x_["bouts"]),
                       self.demographics(x_["demo"]),
                       self.bins(x_["bins"]),
                       self.cosinor(x_["cosinor"]),
                       self.stats(x_["stats"]),
                       self.time(x_["time"]),
                       self.ae24(x_["ae24"]),
                       self.vae24(x_["vae24"]),
                       self.hourly_bins(x_["hourly_bins"]),
                       self.hourly_stats(x_["hourly_stats"]),
                       self.hourly_bouts(x_["hourly_bouts"]),
                       ), 1)
        x = self.drop(x)
        x = self.shared(x)
        x = x.view(x.size(0), -1)

        out = {}
        for metric in self.sleep_metrics:
            out[metric] = self.output[metric](x)

        return out

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3, eps=1e-08)
        print("Current number of parameters:", len(list(self.parameters())))
        optimizer = optim.Adam(self.parameters(), lr=self.opt_learning_rate)

        # optimizer = optim.SGD(self.parameters(), lr=self.opt_learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         patience=self.opt_step_size,
                                                         factor=self.opt_gamma,  # new_lr = lr * factor (default = 0.1)
                                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'loss'}

    def calculate_losses(self, y, predictions):
        loss_fnct = nn.CrossEntropyLoss()

        total_loss = 0
        losses = {}
        losses_list = []

        for metric in self.sleep_metrics:
            loss = loss_fnct(predictions[metric], y[metric])
            losses[metric] = loss

            losses_list.append(loss)

        total_loss = self.loss_aggregator.aggregate_losses(losses_list)
        total_loss = total_loss.type_as(loss)

        return losses, total_loss

    def process_step(self, batch):
        x, y = batch
        x_ = {}
        for fset in self.feature_sets:
            x_[fset] = x[fset].view(x[fset].size(0), -1)

        predictions = self(x_)
        losses, total_loss = self.calculate_losses(y, predictions)
        return predictions, losses, total_loss, y

    def training_step(self, batch, batch_idx):
        predictions, losses, total_loss, y = self.process_step(batch)
        # self.log('loss', total_loss)
        return {'loss': total_loss, 'ys': y, 'preds': predictions, }

    def validation_step(self, batch, batch_idx):
        predictions, losses, total_loss, y = self.process_step(batch)
        # self.log('val_loss', total_loss)
        return {'loss': total_loss, 'ys': y, 'preds': predictions, 'loss_per_metric': losses}

    def test_step(self, batch, batch_idx):
        predictions, losses, total_loss, y = self.process_step(batch)
        return {'loss': total_loss, 'ys': y, 'preds': predictions, 'loss_per_metric': losses}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([row['loss'] for row in outputs]).mean()
        print("(Validation) Total Loss: %.4f" % val_loss)

        avg_metrics = {"acc": [], "macroF1": [], "mcc": [], "loss": []}
        avg_losses = []
        for metric in self.sleep_metrics:
            y = torch.stack([row["ys"][metric] for row in outputs]).view(-1).cpu()
            pred = torch.stack([row['preds'][metric].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()

            avg_loss = torch.stack([row['loss_per_metric'][metric] for row in outputs]).mean()

            acc, macrof1, microf1, mcc = calculate_classification_metrics(y, pred)
            print("(%s) Epoch: %d, ACC: %.3f, MacroF1: %.3f, MCC: %.3f, Loss: %.4f" % (
                metric, self.current_epoch, acc, macrof1, mcc, avg_loss))

            avg_metrics["acc"].append(acc)
            avg_metrics["macroF1"].append(macrof1)
            avg_metrics["mcc"].append(mcc)
            avg_metrics["loss"].append(avg_loss.cpu().numpy())

        if self.loss_method is not None:
            if self.loss_method == "alex":
                print("LOGSIGMAS:", self.loss_aggregator.logsigma)

            elif self.loss_method.startswith("dwa") or self.loss_method.startswith("bandit"):
                self.loss_aggregator.lambda_weight = self.loss_aggregator.lambda_weight.type_as(avg_losses[0])

        self.loss_aggregator.adjust_after_validation(avg_losses, self.current_epoch)

        for eval_metric in ["acc", "macroF1", "mcc", "loss"]:
            avg_metrics[eval_metric] = np.array(avg_metrics[eval_metric]).mean()
            print("Average %s: %.4f" % (eval_metric, avg_metrics[eval_metric]))
            self.log(eval_metric, avg_metrics[eval_metric])

    def test_epoch_end(self, outputs):
        return_dict = {}
        test_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print("(Test) Total Loss: %.4f" % test_loss)

        metric_global = "mcc"
        opt_metric = []
        for metric in self.sleep_metrics:
            y = torch.stack([row["ys"][metric] for row in outputs]).view(-1).cpu()
            pred = torch.stack([row['preds'][metric].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()
            # pred = torch.stack([row['preds'].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()

            avg_loss = torch.stack([row['loss_per_metric'][metric] for row in outputs]).mean()

            acc, macrof1, microf1, mcc = calculate_classification_metrics(y, pred)
            print("(%s) Epoch: %d, ACC: %.3f, MacroF1: %.3f, MCC: %.3f, Loss: %.4f" %
                  (metric, self.current_epoch, acc, macrof1, mcc, avg_loss))

            return_dict["acc_%s" % metric] = acc
            return_dict["macrof1_%s" % metric] = macrof1
            return_dict["mcc_%s" % metric] = mcc
            opt_metric.append({"macrof1": macrof1, "mcc": mcc}[metric_global])

        average_metric = np.array(opt_metric).mean()
        print("Test Average Eval Metric: %.4f" % average_metric)
        return_dict["average_global"] = average_metric
        return_dict["test_loss"] = test_loss

        row = {}
        row["test"] = True
        for p in ["dataset", "loss_method", "opt_gamma", "opt_learning_rate", "sleep_metrics",
                  "dropout_input_layers", "dropout_inner_layers", "opt_step_size", "opt_weight_decay",
                  "output_strategy", "shared_output_size", "batch_size"]:
            row[p] = self.hparams[p]
        for k, v in return_dict.items():
            row[k] = v

        pd.DataFrame([row]).to_csv("nn_results/nn_test_%s.csv.gz" % (self.timestamp), index=False)

        for k, v in return_dict.items():
            self.log(k, v)



if __name__ == "__main__":

    batch_size = 64
    DATASET = "hchs"

    with open("%s_to_NN.pkl" % (DATASET), "rb") as f:
        X, Y = pickle.load(f)

    train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val   = DataLoader(myXYDataset(X["val"],   Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    test  = DataLoader(myXYDataset(X["test"],  Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)


    # +
    # param = [["sleepEfficiency"], 16, 0.005, None, 10, 0.01, 64] # Config for SGD
    # param = [["sleepEfficiency", "awakening", "totalSleepTime", "combined"], 16, 0.005, "dwa_default", 10, 0.01, batch_size] # Config for Adam
    # param = [["sleepEfficiency"], 128, 0.005, None, 10, 0.01, 0.1, 0.1]
    # param = [["sleepEfficiency"], 8, 0.003, None, 10, 0.01, 0.0, 0.05]
    param = [['sleepEfficiency'], 8, 0.0005, 'equal', 5, 0.001, 0.05, 0.1]
    sleep_metrics, shared_output_size, learning_rate, loss_method, opt_step_size, weight_decay, dropout_input_layers, dropout_inner_layets = param

    seed.seed_everything(42)


    path_ckps = "./lightning_logs/test/"

    ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{val_loss:.3f}-{average_global:.3f}",
                          save_top_k=1,
                          verbose=True,
                          # monitor="val_loss", mode="min",
                          monitor="average_global", mode="max",
                          prefix="",
                          )

    early_stop_callback = EarlyStopping(min_delta=0.00,
                                        verbose=True,
                                        monitor='average_global', mode='max',
                                        # monitor='val_loss', mode='min',
                                        patience=10,
                                        )

    hparams = Namespace(batch_size=batch_size,
                        shared_output_size=shared_output_size,
                        sleep_metrics=sleep_metrics,
                        dropout_input_layers=dropout_input_layers,
                        dropout_inner_layets=dropout_inner_layets,
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
                        dataset=DATASET,
                        )

    model = MyNet(hparams)
    model.double()

    trainer = Trainer(gpus=0, min_epochs=2,
                      max_epochs=300,
                      callbacks=[early_stop_callback, ckp],
                      #gradient_clip_val=0.5,
                      )
    trainer.fit(model, train, val)
    trainer.test(test_dataloaders=test)

# LSTM
# (Test) Total Loss: 0.8902
# (sleepEfficiency) Epoch: 13, ACC: 0.789, MacroF1: 0.679, MCC: 0.537, Loss: 0.8902
# Test Average Eval Metric: 0.5367
# (Test) Total Loss: 0.5439
# (sleepEfficiency) Epoch: 14, ACC: 0.797, MacroF1: 0.696, MCC: 0.540, Loss: 0.5439
# Test Average Eval Metric: 0.5404
# (Test) Total Loss: 0.5419
# (sleepEfficiency) Epoch: 12, ACC: 0.783, MacroF1: 0.599, MCC: 0.483, Loss: 0.5419
# Test Average Eval Metric: 0.4826
# (Test) Total Loss: 1.1178
# (sleepEfficiency) Epoch: 19, ACC: 0.800, MacroF1: 0.695, MCC: 0.544, Loss: 1.1178
# Test Average Eval Metric: 0.5438
# (Test) Total Loss: 0.8090
# (sleepEfficiency) Epoch: 18, ACC: 0.781, MacroF1: 0.666, MCC: 0.505, Loss: 0.8090
# Test Average Eval Metric: 0.5047
# (Test) Total Loss: 0.9865
# (sleepEfficiency) Epoch: 20, ACC: 0.777, MacroF1: 0.689, MCC: 0.516, Loss: 0.9865
# Test Average Eval Metric: 0.5164

#### LSTM version 2
# (Test) Total Loss: 1.0198
# (sleepEfficiency) Epoch: 18, ACC: 0.791, MacroF1: 0.631, MCC: 0.532, Loss: 1.0198
# Test Average Eval Metric: 0.5318
# (Test) Total Loss: 0.8892
# (sleepEfficiency) Epoch: 17, ACC: 0.798, MacroF1: 0.698, MCC: 0.543, Loss: 0.8892
# Test Average Eval Metric: 0.5433
# (Test) Total Loss: 0.6308
# (sleepEfficiency) Epoch: 16, ACC: 0.796, MacroF1: 0.708, MCC: 0.542, Loss: 0.6308
# Test Average Eval Metric: 0.5423    
# (Test) Total Loss: 1.0888
# (sleepEfficiency) Epoch: 22, ACC: 0.790, MacroF1: 0.663, MCC: 0.516, Loss: 1.0888
# Test Average Eval Metric: 0.5163

#### Optimized LSTM
# (Test) Total Loss: 0.5369
# (sleepEfficiency) Epoch: 16, ACC: 0.819, MacroF1: 0.733, MCC: 0.583, Loss: 0.5369
# Test Average Eval Metric: 0.5826
# (Test) Total Loss: 0.4813
# (sleepEfficiency) Epoch: 14, ACC: 0.801, MacroF1: 0.696, MCC: 0.577, Loss: 0.4813
# Test Average Eval Metric: 0.5766
# (Test) Total Loss: 0.4959
# (sleepEfficiency) Epoch: 16, ACC: 0.800, MacroF1: 0.711, MCC: 0.558, Loss: 0.4959
# Test Average Eval Metric: 0.5585

## Optimized Normal layers Test
# (Test) Total Loss: 0.4590
# (sleepEfficiency) Epoch: 12, ACC: 0.807, MacroF1: 0.700, MCC: 0.568, Loss: 0.4590
# Test Average Eval Metric: 0.5683
# (Test) Total Loss: 0.5118
# (sleepEfficiency) Epoch: 14, ACC: 0.816, MacroF1: 0.707, MCC: 0.579, Loss: 0.5118
# Test Average Eval Metric: 0.5785
# (Test) Total Loss: 0.5519
# (sleepEfficiency) Epoch: 16, ACC: 0.797, MacroF1: 0.683, MCC: 0.530, Loss: 0.5519
# Test Average Eval Metric: 0.5300
# -



#
# # sleep_metrics, shared_output_size, learning_rate, loss_method, opt_step_size, weight_decay, dropout_input_layers, dropout_inner_layets
# results = eval_n_times(10, [64, ['sleepEfficiency'], 8, 0.0005, 'equal', 5, 0.001, 0.05, 0.1])
# results.mean()
#
# # sleep_metrics, shared_output_size, learning_rate, loss_method, opt_step_size, weight_decay, dropout_input_layers, dropout_inner_layets
# results_equal = eval_n_times(10, [64, ['sleepEfficiency', "totalSleepTime", "awakening", "combined"], 8, 0.0005, 'equal', 5, 0.001, 0.05, 0.1])
# results_equal.mean()
#
# results_alex = eval_n_times(10, [64, ['sleepEfficiency', "totalSleepTime", "awakening", "combined"], 8, 0.0005, 'alex', 5, 0.001, 0.05, 0.1])
# results_alex.mean()
#
# # batch_size, sleep_metrics, shared_output_size, learning_rate, loss_method, opt_step_size, weight_decay, dropout_input_layers, dropout_inner_layets
# results = eval_n_times(50, [128, ['sleepEfficiency'], 32, 0.0005, 'equal', 20, 0.01, 0.0, 0.0])
# results.mean()
#
# # batch_size, sleep_metrics, shared_output_size, learning_rate, loss_method, opt_step_size, weight_decay, dropout_input_layers, dropout_inner_layets
# results2 = eval_n_times(50, [64, ['sleepEfficiency', "totalSleepTime", "awakening", "combined"], 8, 0.0005, 'alex', 5, 0.001, 0.05, 0.1])
# results2.mean()
#
# # batch_size, sleep_metrics, shared_output_size, learning_rate, loss_method, opt_step_size, weight_decay, dropout_input_layers, dropout_inner_layets
# results2 = eval_n_times(50, [16, ['sleepEfficiency'], 8, 0.001, None, 5, 0.01, 0.1, 0.05])
# results2.mean()

# +
# # Runs a lot of experiments
# params = []

# # for sleep_metrics in [["sleepEfficiency", "awakening", "totalSleepTime", "combined"],
# #                          ["sleepEfficiency"], ["awakening"], ["totalSleepTime"], ["combined"]]:
# #     for loss_method in ["dwa_default", "equal", "alex"]:


# for sleep_metrics in [["sleepEfficiency"]]:
#     for loss_method in ["equal"]:
#         for shared_output_size in [8, 16, 32, 64, 128]:
#             for learning_rate in [0.1, 0.01, 0.003, 0.005, 0.001, 0.0005, 0.0001]:
#                 for opt_step_size in [5, 10, 20]:
#                     for weight_decay in [0.01, 0.001]:
#                         for dropout_input_layers in [0.0, 0.05, 0.1]:
#                             for dropout_inner_layets in [0.0, 0.05, 0.1, 0.2]:
#                                 params.append([sleep_metrics, shared_output_size, learning_rate, loss_method, opt_step_size, weight_decay, dropout_input_layers, dropout_inner_layets])

# print("Exploring %d parameter combinations" % len(params))
# DATASET = "hchs"
# with open("%s_to_NN.pkl" % (DATASET), "rb") as f:
#     X, Y = pickle.load(f)

# # idx = int(sys.argv[1])
# batch_sizes = [128] # [8, 16, 32, 64, 128][idx: idx+1]

# for batch_size in batch_sizes:
#     print("BATCH SIZE: %d" % batch_size)
#     train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
#     val   = DataLoader(myXYDataset(X["val"],   Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
#     test  = DataLoader(myXYDataset(X["test"],  Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)

#     for param in params:
#         sleep_metrics, shared_output_size, learning_rate, loss_method, opt_step_size, weight_decay, dropout_input_layers, dropout_inner_layets = param
        
#         path_ckps = "./lightning_logs/sleepEfficiency/"

#         ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{val_loss:.3f}-{average_global:.3f}",
#                       save_top_k=1,
#                       verbose=True,
#                       # monitor="val_loss", mode="min",
#                       monitor="average_global", mode="max",
#                       prefix="",
#                       )

#         early_stop_callback = EarlyStopping(min_delta=0.00,
#                                     verbose=True,
#                                     monitor='average_global', mode='max',
#                                     # monitor='val_loss', mode='min',
#                                     patience=10,
#                                     )

#         hparams = Namespace(batch_size=batch_size,
#                     shared_output_size=shared_output_size,
#                     sleep_metrics=sleep_metrics,
#                     dropout_input_layers=dropout_input_layers,
#                     dropout_inner_layets=dropout_inner_layets,
#                     #
#                     # Optmizer configs
#                     #
#                     opt_learning_rate=learning_rate,
#                     opt_weight_decay=weight_decay,
#                     opt_step_size=opt_step_size,
#                     opt_gamma=0.5,
#                     #
#                     # Loss combination method
#                     #
#                     loss_method=loss_method,  # Options: equal, alex, dwa
#                     #
#                     # Output layer
#                     #
#                     output_strategy="linear",  # Options: attention, linear
#                     dataset=DATASET,
#                     )

#         model = MyNet(hparams)
#         model.double()

#         trainer = Trainer(gpus=1, min_epochs=2, max_epochs=100,
#                   callbacks=[early_stop_callback, ckp],)
#         trainer.fit(model, train, val)
#         trainer.test(test_dataloaders=test)






