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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from collections import OrderedDict
from datetime import datetime

from MTLModels import MTLRandom, MTLEqual, MTLBandit, MTLDWA


# -

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

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
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

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        return x

class Net8Input(pl.LightningModule):
    # Used for stats, time

    def __init__(self,
                 input_size=8,
                 output_dim=2,
                 dropout_rate=0.0,
                 ):
        super(Net8Input, self).__init__()

        self.lin1 = nn.Linear(input_size, 4)
        self.lin2 = nn.Linear(4, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        return x

class Net16Input(pl.LightningModule):
    # 
    def __init__(self,
                 input_size=16,
                 output_dim=2,
                 dropout_rate=0.0,
                 ):
        super(Net16Input, self).__init__()

        self.lin1 = nn.Linear(input_size, 8)
        self.lin2 = nn.Linear(8, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
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

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
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
        self.lin3 = nn.Linear(32, 8)
        self.lin4 = nn.Linear(8, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.act(self.lin3(x))
        x = self.act(self.lin4(x))
        return x



# +
class MTL:
    def __init__(self):
        pass

    def aggregate_losses(self, losses):
        pass

    def adjust_after_validation(self, losses, epoch):
        pass


class MTLEqual(MTL):

    def __init__(self, ntasks):
        super(MTLEqual, self).__init__()
        self.ntasks = ntasks

    def aggregate_losses(self, losses):
        return sum(losses) / self.ntasks

    def adjust_after_validation(self, losses, epoch):
        return


def calculate_classification_metrics(labels, predictions):
    return metrics.accuracy_score(labels, predictions),\
           metrics.f1_score(labels, predictions, average='macro', labels=[0, 1, 2]), \
           metrics.f1_score(labels, predictions, average='micro', labels=[0, 1, 2]), \
           metrics.matthews_corrcoef(labels, predictions)


# -

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


# +
class MyNet(pl.LightningModule):

    def __init__(self,
                 hparams,
                 X, Y
                 ):

        super(MyNet, self).__init__()

        self.hparams = hparams
        self.X = X
        self.Y = Y

        self.save_hyperparameters()
        
        saved_results = None

        # Optimizer configs
        self.opt_learning_rate = hparams.opt_learning_rate
        self.opt_weight_decay = hparams.opt_weight_decay
        self.opt_step_size = hparams.opt_step_size
        self.opt_gamma = hparams.opt_gamma

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

        output_sizes = {"bouts": 2, "demo":8, "bins": 1, "cosinor": 1, "stats": 1, "time": 1,
                        "ae24":1,"vae24":1, "hourly_bins":4, "hourly_stats":4, "hourly_bouts":4}
        
        self.demographics = NetDemographics(input_size=get_input_size(self.dataset, "demo"), output_dim=output_sizes["demo"])

        self.bouts = Net16Input(input_size=get_input_size(self.dataset, "bouts"), output_dim=output_sizes["bouts"])

        self.stats = Net8Input(input_size=get_input_size(self.dataset, "stats"), output_dim=output_sizes["stats"])
        self.time = Net8Input(input_size=get_input_size(self.dataset, "time"), output_dim=output_sizes["time"])

        self.ae24 = Net4Input(input_size=get_input_size(self.dataset, "ae24"), output_dim=output_sizes["ae24"])
        self.vae24 = Net4Input(input_size=get_input_size(self.dataset, "vae24"), output_dim=output_sizes["vae24"])
        self.cosinor = Net4Input(input_size=get_input_size(self.dataset, "cosinor"), output_dim=output_sizes["cosinor"])
        self.bins = Net4Input(input_size=get_input_size(self.dataset, "bins"), output_dim=output_sizes["bins"])

        self.hourly_bins = Net64Input(input_size=get_input_size(self.dataset, "hourly_bins"), output_dim=output_sizes["hourly_bins"])
        self.hourly_stats = Net128Input(input_size=get_input_size(self.dataset, "hourly_stats"), output_dim=output_sizes["hourly_stats"])
        self.hourly_bouts = Net128Input(input_size=get_input_size(self.dataset, "hourly_bouts"), output_dim=output_sizes["hourly_bouts"])

        # self.sleep_qualities = NetSleepQualities(output_dim=self.internal_outputs)
        # self.activitiy_pa2d  = NetActGrid((2,3,24), output_dim=self.internal_outputs)
        # self.activitiy_grid  = NetActGrid((1,3,12), output_dim=self.internal_outputs)
        # self.time            = NetTimeFeatures(output_dim=self.internal_outputs)

        self.shared = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(sum(output_sizes.values()), 16)),
            ('act1', nn.ReLU(inplace=True)),
            ('lin2', nn.Linear(16, self.shared_output_size)),
            ('act2', nn.ReLU(inplace=True)),
        ]))

#         self.output = nn.ModuleDict()
#         for metric in self.sleep_metrics:
#             if hparams.output_strategy == "linear":
#                 self.output[metric] = nn.Sequential(nn.Linear(self.shared_output_size, 3),
#                                                     #nn.ELU(inplace=True),
#                                                     #nn.Linear(8, 3),
#                                                     # nn.ELU(inplace=True),
#                                                     # nn.Linear(4, 3)
#                                                     )
        self.output = nn.Linear(self.shared_output_size, 3)

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


        if saved_results is None:
            self.saved_results = []
        else:
            self.saved_results = saved_results

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

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
        x = self.shared(x)
        x = x.view(self.batch_size, x.shape[-1])

#         out = {}
#         for metric in self.sleep_metrics:
#             out[metric] = self.output[metric](x)  # .view(self.batch_size * 3)
        out = self.output(x)
    
        return out

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3, eps=1e-08)
        print("Current number of parameters:", len(list(self.parameters())))
        optimizer = optim.Adam(self.parameters(), lr=self.opt_learning_rate)
        return optimizer       
#         optimizer = optim.AdamW(self.parameters(),
#                                 lr=self.opt_learning_rate,
#                                 weight_decay=self.opt_weight_decay)
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opt_step_size, gamma=self.opt_gamma)
#        return [optimizer], [scheduler]

    def prepare_data(self):
        self.train_dataset = myXYDataset(self.X["train"], self.Y["train"])
        self.val_dataset = myXYDataset(self.X["val"], self.Y["val"])
        self.test_dataset = myXYDataset(self.X["test"], self.Y["test"])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8)

    def calculate_losses(self, y, predictions):
        loss_fnct = nn.CrossEntropyLoss()

        total_loss = 0
        losses = {}
        losses_list = []

        for metric in self.sleep_metrics:
            #loss = loss_fnct(predictions[metric], y[metric])
            loss = loss_fnct(predictions, y[metric])
            losses[metric] = loss

            losses_list.append(loss)
        
        total_loss = self.loss_aggregator.aggregate_losses(losses_list)
        total_loss = total_loss.type_as(loss)

        return losses, total_loss

    def process_step(self, batch):
        x, y = batch
        x_ = {}
        for fset in self.feature_sets:
            x_[fset] = x[fset].view(self.batch_size, get_input_size(self.dataset, fset))

        predictions = self(x_)
        losses, total_loss = self.calculate_losses(y, predictions)
        return predictions, losses, total_loss, y

    def training_step(self, batch, batch_idx):
        _, _, total_loss, _ = self.process_step(batch)
        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        predictions, losses, total_loss, y = self.process_step(batch)
        return {'val_tmp_loss': total_loss, 'ys': y, 'preds': predictions, 'loss_per_metric': losses}

    def test_step(self, batch, batch_idx):
        predictions, losses, total_loss, y = self.process_step(batch)
        return {'test_tmp_loss': total_loss, 'ys': y, 'preds': predictions, 'loss_per_metric': losses}

    def validation_epoch_end(self, outputs):
        return_dict = {}
        val_loss = torch.stack([row['val_tmp_loss'] for row in outputs]).mean()
        print("Total Loss: %.4f" % val_loss)

        metric_global = "mcc"

        opt_metric = []
        avg_losses = []
        for metric in self.sleep_metrics:
            y = torch.stack([row["ys"][metric] for row in outputs]).view(-1).cpu()
            #pred = torch.stack([row['preds'][metric].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()
            pred = torch.stack([row['preds'].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()

            avg_loss = torch.stack([row['loss_per_metric'][metric] for row in outputs]).mean()
            avg_losses.append(avg_loss)

            acc, macrof1, microf1, mcc = calculate_classification_metrics(y, pred)
            print("(%s) Epoch: %d, ACC: %.3f, MacroF1: %.3f, MCC: %.3f, Loss: %.4f" % (
                metric, self.current_epoch, acc, macrof1, mcc, avg_loss))

            return_dict["acc_%s" % metric] = acc
            return_dict["macrof1_%s" % metric] = macrof1
            return_dict["mcc_%s" % metric] = mcc
            opt_metric.append({"macrof1": macrof1, "mcc":mcc}[metric_global])

        if self.loss_method is not None:
            if self.loss_method == "alex":
                print("LOGSIGMAS:", self.loss_aggregator.logsigma)

            elif self.loss_method.startswith("dwa") or self.loss_method.startswith("bandit"):
                self.loss_aggregator.lambda_weight = self.loss_aggregator.lambda_weight.type_as(avg_losses[0])

        self.loss_aggregator.adjust_after_validation(avg_losses, self.current_epoch)

        average_metric = np.array(opt_metric).mean()
        print("Average Eval Metric: %.4f" % average_metric)
        return_dict["val_loss"] = val_loss
        return_dict["average_global"] = average_metric
        #return_dict["metric_global"] = metric_global
        logs = return_dict.copy()
        return_dict["log"] = logs
        return_dict["progress_bar"] = {"val_loss": val_loss}

        
        return_dict["test"] = False
        for p in ["batch_size", "dataset", "loss_method", "opt_gamma", "opt_learning_rate",
                 "opt_step_size", "opt_weight_decay", "output_strategy", "shared_output_size"]:
            return_dict[p] = self.hparams[p]
        
        # Save test results to a file
        self.saved_results.append(return_dict)
        print("Saving results to disk!")
        
        timestamp = datetime.now()
        pd.DataFrame(self.saved_results).to_csv("nn_results/nn_val_%s.out" % (timestamp), index=False)
        
        return return_dict

    def test_epoch_end(self, outputs):
        return_dict = {}
        test_loss = torch.stack([x['test_tmp_loss'] for x in outputs]).mean()
        print("Total Test Loss: %.4f" % test_loss)

        metric_global = "mcc"
        opt_metric = []
        for metric in self.sleep_metrics:
            y = torch.stack([row["ys"][metric] for row in outputs]).view(-1).cpu()
            #pred = torch.stack([row['preds'][metric].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()
            pred = torch.stack([row['preds'].max(1, keepdim=True)[1] for row in outputs]).view(-1).cpu()

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
        #return_dict["metric_global"] = metric_global
        return_dict["test_loss"] = test_loss.item()
        return_dict["sleep_metrics"] = self.sleep_metrics

        return_dict["test"] = True
        for p in ["batch_size", "dataset", "loss_method", "opt_gamma", "opt_learning_rate",
                 "opt_step_size", "opt_weight_decay", "output_strategy", "shared_output_size"]:
            return_dict[p] = self.hparams[p]
        
        # Save test results to a file
        self.saved_results.append(return_dict)
        print("Saving results to disk!")
        
        timestamp = datetime.now()
        pd.DataFrame(self.saved_results).to_csv("nn_results/nn_results_%s.out" % (timestamp), index=False)

        return return_dict


# +
path_ckps = "./lightning_logs/test/"

ckp = ModelCheckpoint(filepath=path_ckps + "{epoch:03d}-{val_loss:.3f}-{average_global:.3f}",
                      save_top_k=1,
                      verbose=True,
                      monitor="average_global",  # "average_r2",
                      mode="max",
                      prefix="",
                      )

early_stop_callback = EarlyStopping(monitor='average_global',  # "average_r2",
                                    min_delta=0.00,
                                    patience=50,
                                    verbose=True,
                                    mode='max'
                                    )
# +
param = [["sleepEfficiency"], 8, 0.001, None, 20, 0.01, 8]
sleep_metrics, shared_output_size, learning_rate, loss_method, opt_step_size, weight_decay, batch_size = param

hparams = Namespace(batch_size=batch_size,
                    shared_output_size=shared_output_size,
                    sleep_metrics=sleep_metrics,
                    # Optmizer configs
                    opt_learning_rate=learning_rate,
                    opt_weight_decay=weight_decay,
                    opt_step_size=opt_step_size,
                    opt_gamma=0.1,
                    # Loss combination method
                    loss_method=loss_method,  # Options: equal, alex, dwa
                    # Output layer
                    output_strategy="linear",  # Options: attention, linear
                    dataset=DATASET,
                    )


with open("%s_to_NN.pkl" % (DATASET), "rb") as f:
    X, Y = pickle.load(f)

model = MyNet(hparams, X, Y)
model.double()

trainer = Trainer(gpus=1, min_epochs=10, max_epochs=300,
                  callbacks=[early_stop_callback], # ckp],
                  gradient_clip_val=0.5,
                  )
trainer.fit(model)
trainer.test()

#Total Loss: 2.9943
#(sleepEfficiency) Epoch: 223, ACC: 0.790, MacroF1: 0.657, MCC: 0.514, Loss: 2.9943
#Average Eval Metric: 0.5145

# +
# Run a lot of experiments
params = []
for sleep_metrics in [["sleepEfficiency", "awakening", "totalSleepTime", "combined"],
                         ["sleepEfficiency"], ["awakening"], ["totalSleepTime"], ["combined"]]:
    for shared_output_size in [4, 8, 16]:
        for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
            for loss_method in ["dwa_default", "equal", "alex"]:
                for opt_step_size in [10, 20]:
                    for weight_decay in [0.01, 0.001]:
                        for batch_size in [8, 16, 32]:
                            params.append([sleep_metrics, shared_output_size, learning_rate, loss_method, opt_step_size, weight_decay, batch_size])

DATASET = "hchs"
for param in params[:]:
    
    sleep_metrics, shared_output_size, learning_rate, loss_method, opt_step_size, weight_decay, batch_size = param

    hparams = Namespace(batch_size=batch_size,
                        shared_output_size=shared_output_size,
                        sleep_metrics=sleep_metrics,
                        # Optmizer configs
                        opt_learning_rate=learning_rate,
                        opt_weight_decay=weight_decay,
                        opt_step_size=opt_step_size,
                        opt_gamma=0.1,
                        # Loss combination method
                        loss_method=loss_method,  # Options: equal, alex, dwa
                        # Output layer
                        output_strategy="linear",  # Options: attention, linear
                        dataset=DATASET,
                        )


    with open("%s_to_NN.pkl" % (DATASET), "rb") as f:
        X, Y = pickle.load(f)

    model = MyNet(hparams, X, Y)
    model.double()

    trainer = Trainer(gpus=1, min_epochs=1, max_epochs=2,
                      callbacks=[early_stop_callback, ckp],
                      gradient_clip_val=0.0,
                      )
    trainer.fit(model)
    trainer.test()


# +
# from pytorch_lightning.callbacks import Callback

# class PandasCallback(Callback):
#     def __init__(self, temp_path):
#         super(Callback, self).__init__()
#         self.temp_path = temp_path

#     def on_test_end(self, trainer, pl_module):
#         # Alternative is using trainer.callback_metrics
#         pd.DataFrame(pl_module.saved_results).to_csv(self.temp_path, index=False)

# class IgnoreLogger(LightningLoggerBase):
#     """PyTorch Lightning `dict` logger."""

#     def __init__(self):
#         super(IgnoreLogger, self).__init__()

#     def log_metrics(self, metrics, step=None):
#         pass

#     @rank_zero_only
#     def log_hyperparams(self, params):
#         pass

#     @property
#     def version(self):
#         return 0

#     def experiment(self):
#         pass

#     def name(self):
#         pass


# +
# # Run models from datapath
# import os
# from glob import glob

# output_file = "final_results.csv"
# DATASET = "hchs"

# with open("%s_to_NN.pkl" % (DATASET), "rb") as f:
#     X, Y = pickle.load(f)

# ignorelogger = IgnoreLogger()
# ignorecheckpoint = pl.callbacks.ModelCheckpoint(
#     os.path.join("./", "ignore_{}".format(0)), monitor="val_loss"
# )

# checkpoints = glob(path_ckps + "/*")

# print("RUNNING TESTS WITH BEST MODELS")
# for i, ckp in enumerate(checkpoints):
#     print(ckp)
#     m = MyNet.load_from_checkpoint(ckp, X=X, Y=Y, hparam=hparams)
#     m.double()
#     m.prepare_data()
#     trainer = Trainer(gpus=1,
#                      #logger=ignorelogger,
#                      #checkpoint_callback=ignorecheckpoint,
#                      callbacks = [PandasCallback("./.tmp_results%d.csv" % (i))],
#                      resume_from_checkpoint=ckp
#     )
#     #trainer = Trainer(gpus=1, )
#     trainer.test(m)

# tmp_files = glob("./.tmp_results*.csv")

# data_frames = []
# for f in tmp_files:
#     data_frames.append(pd.read_csv(f))
#     os.remove(f)

# pd.concat(data_frames).to_csv(output_file, index=False)
# print("Done! Saved file %s." % (output_file))

# -


