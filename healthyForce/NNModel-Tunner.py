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


# -

def get_data(DATASET="hchs"):
    fullpath = "/home/palotti/github/HypnosPy/healthyForce/"
    # fullpath = "/export/sc2/jpalotti/github/HypnosPy/healthyForce/"
    with open(fullpath + "%s_to_NN.pkl" % (DATASET), "rb") as f:
        X, Y = pickle.load(f)
    return X, Y


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
    return metrics.accuracy_score(labels, predictions),\
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
        X = self.drop(x)
        x = self.act(self.lin1(x))
        X = self.drop(x)
        x = self.act(self.lin2(x))
        X = self.drop(x)
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
        X = self.drop(x)
        x = self.act(self.lin1(x))
        X = self.drop(x)
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
        X = self.drop(x)
        x = self.act(self.lin1(x))
        X = self.drop(x)
        x = self.act(self.lin2(x))
        X = self.drop(x)
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
        X = self.drop(x)
        x = self.act(self.lin1(x))
        X = self.drop(x)
        x = self.act(self.lin2(x))
        X = self.drop(x)
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
        X = self.drop(x)
        x = self.act(self.lin1(x))
        X = self.drop(x)
        x = self.act(self.lin2(x))
        X = self.drop(x)
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
        self.lin3 = nn.Linear(32, output_dim) # 8)
        #self.lin4 = nn.Linear(8, output_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        X = self.drop(x)
        x = self.act(self.lin1(x))
        X = self.drop(x)
        x = self.act(self.lin2(x))
        X = self.drop(x)
        x = self.act(self.lin3(x))
        #x = self.act(self.lin4(x))
        return x
   
    
def get_number_internal_layers(n, output_size):
    """
    E.g.:
        get_number_internal_layers(20, 3) --> [16, 8, 4]
        get_number_internal_layers(192, 16) # --> [128, 64, 32]
    """
    i = 1; d = 2; s = []
    while (n - 1) / d > 1: 
        s.append(d)
        i += 1
        d = 2**i;
        
    s = [e for e in s if e > output_size]
    return s[::-1]


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
        
        
        self.lstm = nn.LSTM(break_point, hidden_dim, num_layers=num_layers, dropout=dropout_rate,
                            batch_first=True, bidirectional=bidirectional)
        self.linlayers = nn.ModuleList() 
        
        if bidirectional:
            hidden_dim *= 2
        
        last_d = hidden_dim * (input_size//break_point)
        for lay_size in get_number_internal_layers(last_d, output_dim):
            print("Last: %d, Next: %d" % (last_d, lay_size))
            self.linlayers.append(nn.Linear(last_d, lay_size))
            last_d = lay_size
            
        print("Very Last: %d, Out: %d" % (last_d, output_dim))
        print("#Lin layers: ", len(self.linlayers))
        self.last_lin  = nn.Linear(last_d, output_dim)
        self.act = nn.ReLU()
        self.break_point = break_point
        
    def forward(self, x):
        # print("INPUT:", x.shape)
        x = x.view(x.shape[0], x.shape[1]//self.break_point, -1)
        # print("Reshaped to:", x.shape)
        
        x, hidden = self.lstm(x)
        # print("After LSTM:", x.shape)
        
        #x = x.squeeze()
        x = x.reshape(x.shape[0], -1)
        
        #print("After reshape:", x.shape)
        for lay in self.linlayers:
            x = self.act(lay(x))
        
        x = self.act(self.last_lin(x))
        return x


# -

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
                             "stats", "time", "ae24","vae24",
                             "hourly_bins", "hourly_stats", "hourly_bouts"
                            ]

        if list(self.structure.keys())[0] == "lstm":
            self.net = LSTMLayer(input_size=1232, break_point=1232,
                                 dropout_rate=self.dropout_input_layers,
                                 hidden_dim=self.structure["lstm"]["hidden_dim"],
                                 bidirectional=self.structure["lstm"]["bidirectional"],
                                 num_layers=self.structure["lstm"]["num_layers"],
                                 output_dim=256,
                                )
        
        else:
            self.net = nn.Sequential(OrderedDict([
                            ('lin1', nn.Linear(1232, 512)),
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
            #self.loss_aggregator.logsigma = self.loss_aggregator.logsigma.to("cuda:0")

            self.mylogsigma = nn.Parameter(torch.zeros(len(self.sleep_metrics)))
            # Link parameters to MTLUncertanty model:
            self.loss_aggregator.logsigma = self.mylogsigma
            #self.logsigma = nn.Parameter(torch.zeros(len(self.sleep_metrics)))

        elif self.loss_method.startswith("dwa"):
            self.loss_aggregator = MTLDWA(len(self.sleep_metrics), self.loss_method.split("_")[1])

        elif "bandit" in self.loss_method:
            self.loss_aggregator = MTLBandit(len(self.sleep_metrics),
                                            strategy=self.loss_method,)

        # Save the results every epoch
        self.saved_results = []
        
    def forward(self, x_):
        x = torch.cat((x_["bouts"], x_["demo"], x_["bins"], x_["cosinor"], x_["stats"], x_["time"],
                       x_["ae24"], x_["vae24"], x_["hourly_bins"], x_["hourly_stats"], x_["hourly_bouts"]), 1)
        #print("SHAPE:", x.shape)
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min',
                                                         patience=self.opt_step_size,
                                                         factor=self.opt_gamma, # new_lr = lr * factor (default = 0.1)
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
        #self.log('loss', total_loss)
        return {'loss': total_loss, 'ys': y, 'preds': predictions,}
        
    def validation_step(self, batch, batch_idx):
        predictions, losses, total_loss, y = self.process_step(batch)
        #self.log('val_loss', total_loss)
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
            #pred = torch.stack([row['preds'].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()

            avg_loss = torch.stack([row['loss_per_metric'][metric] for row in outputs]).mean()

            acc, macrof1, microf1, mcc = calculate_classification_metrics(y, pred)
            print("(%s) Epoch: %d, ACC: %.3f, MacroF1: %.3f, MCC: %.3f, Loss: %.4f" %
                  (metric, self.current_epoch, acc, macrof1, mcc, avg_loss))

            return_dict["acc_%s" % metric] = acc
            return_dict["macrof1_%s" % metric] = macrof1
            return_dict["mcc_%s" % metric] = mcc
            opt_metric.append({"macrof1": macrof1, "mcc":mcc}[metric_global])

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
             "hourly_bouts": 372, "time": 8, "cosinor": 3, "demo": 200, "ae_bouts": 8, "ae24": 4,
             "vae24": 4,
            },
            "hchs":
            {"bins": 4, "hourly_bins": 96, "stats": 8, "hourly_stats": 192, "bouts": 16,
             "hourly_bouts": 380, "time": 8,  "cosinor": 3, "demo": 517, "ae_bouts": 8, "ae24": 4,
             "vae24": 4
            }
           }[dataset][fset]


class MyTwoStepsNet(pl.LightningModule):

    def __init__(self,
                 hparams,
                 #X, Y
                 ):

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
        self.structure = hparams.structure

        # Other configs
        self.dataset = hparams.dataset
        self.batch_size = hparams.batch_size
        self.shared_output_size = hparams.shared_output_size
        self.sleep_metrics = hparams.sleep_metrics
        self.loss_method = hparams.loss_method

        self.feature_sets = ["bouts", "demo", "bins", "cosinor",
                             "stats", "time", "ae24","vae24",
                             "hourly_bins", "hourly_stats", "hourly_bouts"
                            ]

        output_sizes = {"bouts": 2, "demo": 16, "bins": 2, "cosinor": 2, "stats": 4, "time": 2,
                        "ae24":2, "vae24":2, "hourly_bins":8, "hourly_stats":16, "hourly_bouts":16}

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
                                         hidden_dim=get_input_size(self.dataset, "hourly_bins")//2,
                                         output_dim=output_sizes["hourly_bins"], dropout_rate=self.dropout_input_layers)
            self.hourly_stats = LSTMLayer(input_size=get_input_size(self.dataset, "hourly_stats"), hidden_dim=get_input_size(self.dataset, "hourly_stats")//2,
                                            output_dim=output_sizes["hourly_stats"], dropout_rate=self.dropout_input_layers)
            self.hourly_bouts = LSTMLayer(input_size=get_input_size(self.dataset, "hourly_bouts"), hidden_dim=get_input_size(self.dataset, "hourly_bouts")//2,
                                           output_dim=output_sizes["hourly_bouts"], dropout_rate=self.dropout_input_layers,
                                         break_point=get_input_size(self.dataset, "hourly_bouts"))
        else:
            self.hourly_bins = Net64Input(input_size=get_input_size(self.dataset, "hourly_bins"), 
                                          output_dim=output_sizes["hourly_bins"], dropout_rate=self.dropout_input_layers)
            self.hourly_stats = Net128Input(input_size=get_input_size(self.dataset, "hourly_stats"), 
                                            output_dim=output_sizes["hourly_stats"], dropout_rate=self.dropout_input_layers)
            self.hourly_bouts = Net128Input(input_size=get_input_size(self.dataset, "hourly_bouts"),
                                            output_dim=output_sizes["hourly_bouts"], dropout_rate=self.dropout_input_layers)
        
        
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
            #self.loss_aggregator.logsigma = self.loss_aggregator.logsigma.to("cuda:0")

            self.mylogsigma = nn.Parameter(torch.zeros(len(self.sleep_metrics)))
            # Link parameters to MTLUncertanty model:
            self.loss_aggregator.logsigma = self.mylogsigma
            #self.logsigma = nn.Parameter(torch.zeros(len(self.sleep_metrics)))

        elif self.loss_method.startswith("dwa"):
            self.loss_aggregator = MTLDWA(len(self.sleep_metrics), self.loss_method.split("_")[1])

        elif "bandit" in self.loss_method:
            self.loss_aggregator = MTLBandit(len(self.sleep_metrics),
                                            strategy=self.loss_method,)

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

        #optimizer = optim.SGD(self.parameters(), lr=self.opt_learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min',
                                                         patience=self.opt_step_size,
                                                         factor=self.opt_gamma, # new_lr = lr * factor (default = 0.1)
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
        #self.log('loss', total_loss)
        return {'loss': total_loss, 'ys': y, 'preds': predictions,}
        
    def validation_step(self, batch, batch_idx):
        predictions, losses, total_loss, y = self.process_step(batch)
        #self.log('val_loss', total_loss)
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
            #pred = torch.stack([row['preds'].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()

            avg_loss = torch.stack([row['loss_per_metric'][metric] for row in outputs]).mean()

            acc, macrof1, microf1, mcc = calculate_classification_metrics(y, pred)
            print("(%s) Epoch: %d, ACC: %.3f, MacroF1: %.3f, MCC: %.3f, Loss: %.4f" %
                  (metric, self.current_epoch, acc, macrof1, mcc, avg_loss))

            return_dict["acc_%s" % metric] = acc
            return_dict["macrof1_%s" % metric] = macrof1
            return_dict["mcc_%s" % metric] = mcc
            opt_metric.append({"macrof1": macrof1, "mcc":mcc}[metric_global])

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

# -

def eval_n_times(config, NetClass, n, gpus=1):
    
    dataset = "hchs"
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    
    if "structure" in config and config["structure"] == "mpl":
        structure = {"mpl": "mpl"}
    else:
        structure = {"lstm": {"bidirectional": config["bidirectional"], "hidden_dim": config["hidden_dim"], "num_layers": config["num_layers"]}}
    monitor = config["monitor"]
    shared_output_size = config["shared_output_size"]
    opt_step_size = config["opt_step_size"]
    weight_decay = config["weight_decay"]
    dropout_input_layers = config["dropout_input_layers"]
    dropout_inner_layers = config["dropout_inner_layers"]
    
    if "sleep_metrics" in config:
        sleep_metrics = config['sleep_metrics']
    else:
        sleep_metrics = ['sleepEfficiency']
    loss_method = 'equal'
    

    X, Y = get_data(dataset)
    train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val   = DataLoader(myXYDataset(X["val"],   Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    test  = DataLoader(myXYDataset(X["test"],  Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    
    results = []
    for s in range(n):
        seed.seed_everything(s)

        path_ckps = "./lightning_logs/test/"

        if monitor == "mcc":
            early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='mcc', mode='max', patience=3)
            ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False, prefix="",
                                  monitor="mcc", mode="max")
        else:
            early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min', patience=3)
            ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False, prefix="",
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


def hyper_tuner(config, NetClass, dataset):
    
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    
    if "structure" in config and config["structure"] == "mpl":
        structure = {"mpl": "mpl"}
    else:
        structure = {"lstm": {"bidirectional": config["bidirectional"], "hidden_dim": config["hidden_dim"], "num_layers": config["num_layers"]}}

    monitor = config["monitor"]
    shared_output_size = config["shared_output_size"]
    opt_step_size = config["opt_step_size"]
    weight_decay = config["weight_decay"]
    dropout_input_layers = config["dropout_input_layers"]
    dropout_inner_layers = config["dropout_inner_layers"]
    
    sleep_metrics = ['sleepEfficiency']
    loss_method = 'equal'
    
    X, Y = get_data(dataset)
    
    train = DataLoader(myXYDataset(X["train"], Y["train"]), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val   = DataLoader(myXYDataset(X["val"],   Y["val"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    test  = DataLoader(myXYDataset(X["test"],  Y["test"]), batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
    
    results = []
    
    seed.seed_everything(42)

    path_ckps = "./lightning_logs/test/"

    if monitor == "mcc":
        early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='mcc', mode='max', patience=3)
        ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False, prefix="",
                              monitor="mcc", mode="max")
    else:
        early_stop_callback = EarlyStopping(min_delta=0.00, verbose=False, monitor='loss', mode='min', patience=3)
        ckp = ModelCheckpoint(filename=path_ckps + "{epoch:03d}-{loss:.3f}-{mcc:.3f}", save_top_k=1, verbose=False, prefix="",
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
    
    trainer = Trainer(gpus=0, min_epochs=2, max_epochs=100,
                      callbacks=[early_stop_callback, ckp, tune_cb])
    trainer.fit(model, train, val)


def run_tuning_procedure(config, expname, ntrials, ncpus, ngpus, NetClass, dataset="hchs"):

    trainable = tune.with_parameters(hyper_tuner, NetClass=NetClass, dataset=dataset)

    analysis = tune.run(trainable,
                        resources_per_trial={"cpu": ncpus, "gpu": ngpus},
                        metric="loss",
                        mode="min",
                        config=config,
                        num_samples=ntrials,
                        name=expname)

    print("Best Parameters:", analysis.best_config)

    analysis.best_result_df.to_csv("best_parameters_exp%s_trials%d.csv" % (expname, ntrials))
    analysis.results_df.to_csv("all_results_exp%s_trials%d.csv" % (expname, ntrials))
    print("Best 5 results")
    print(analysis.results_df.sort_values(by="mcc", ascending=False).head(5))



# +
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

# +
ncpus=12
ngpus=1
ntrials=2
exp_idx = 0 # sys.argv[1]

experiment_list = [
    [MyNet,         default_mpl,  "MyNetMPL"], 
    [MyTwoStepsNet, default_mpl,  "MyTwoStepsNetMPL"],
    [MyNet,         default_lstm, "MyNetLSTM"],
    [MyTwoStepsNet, default_lstm, "MyTwoStepsNetLSTM"],
]

NetClass = experiment_list[exp_idx][0]
config = experiment_list[exp_idx][1]
exp_name = experiment_list[exp_idx][2]

run_tuning_procedure(config, exp_name, ntrials=ntrials, ncpus=ncpus, ngpus=ngpus, NetClass=NetClass, dataset="hchs")

# -

# # Experiments with best parameters

# config_lstm = {'learning_rate': 0.001532635835186596, 'batch_size': 128, 'bidirectional': True, 'num_layers': 2, 'monitor': 'mcc', 'shared_output_size': 64, 'opt_step_size': 5
# , 'weight_decay': 0.0002774925005331883, 'dropout_input_layers': 0.9030694535320151, 'dropout_inner_layers': 0.6805005401772247, 'hidden_dim': 128}
# results2 = eval_n_times(config_lstm, MyNet, 10, gpus=1)


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
