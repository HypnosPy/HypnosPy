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
from glob import glob
import numpy as np
import scipy

from sklearn import metrics
from sklearn import dummy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor, DummyClassifier

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from torch.utils.data import DataLoader

from skorch import NeuralNet, NeuralNetRegressor


# +
df = pd.read_csv("hchs/HCHS_per_hour.csv",  converters={"raw_pa": lambda x: np.fromstring(x, sep=',')})
#df = pd.read_csv("mesa/MESA_per_hour.csv",  converters={"raw_pa": lambda x: np.fromstring(x, sep=',')})

rawpa = []
for m in range(0, 60):
    for s in (0, 30):
        rawpa.append("hyp_act_x_%s_%s" % (str(m).zfill(2), str(s).zfill(2)))

df = df[["pid", "ml_sequence", "hyp_time_col", *rawpa]].fillna(0.0).drop_duplicates()

df = df.pivot(["pid", "ml_sequence"], columns=["hyp_time_col"])
df.columns = df.columns.swaplevel(0, 1)
df.sort_index(axis=1, level=0, inplace=True)


# +
### DF hours
dfs = []
for hour in range(0,24):
    dfs.append(df[hour].mean(axis=1))

df_hour = pd.concat(dfs, axis=1)


# -

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(24*120, 80),
            nn.ReLU(True),
            nn.Linear(80, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 80),
            nn.ReLU(True), nn.Linear(80, 120*24), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# +
# ===== Encoders for 2280-D ========#
class LinearEncoder2880(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(2880, 1440), nn.ReLU(),
            nn.Linear(1440, 360), nn.ReLU(),
            nn.Linear(360, 90), nn.ReLU(),
            nn.Linear(90, 30), nn.ReLU(),
        )
            
    def forward(self, X):
        encoded = self.encode(X)
        return encoded
    
class LinearDecoder2880(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decode = nn.Sequential(
            nn.Linear(30, 90), nn.ReLU(),
            nn.Linear(90, 360), nn.ReLU(),
            nn.Linear(360, 1440), nn.ReLU(),
            nn.Linear(1440, 2880),
        )
        
    def forward(self, X):
        decoded = self.decode(X)
        return decoded
    
    
class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Expected input to CNN is (Batch, Channels, L)
        self.encode = nn.Sequential(
            # input: b, 1, 2880
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=20, stride=10, padding=0), nn.ReLU(),
            #nn.MaxPool1d(2, stride=2), # b, 4, 48
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=6, stride=4, padding=0), nn.ReLU(),
            #nn.MaxPool1d(2, stride=2), # b, 8, 5
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(568, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
         )
            
    def forward(self, X):
        X = X.unsqueeze(1)
        encoded = self.encode(X)
        return encoded

class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decode = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 568), nn.ReLU(),
            nn.Unflatten(1, torch.Size([8, 71])),
            nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=6, stride=4, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size=30, stride=10, padding=0),
            nn.ReLU()
        )
        
    def forward(self, X):
        decoded = self.decode(X)
        return decoded.squeeze(1)
    
# ===== Encoders for 24-D ========#

class LinearEncoder24(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(24, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
            
    def forward(self, X):
        encoded = self.encode(X)
        return encoded
    
class LinearDecoder24(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decode = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 24)
        )
        
    def forward(self, X):
        decoded = self.decode(X)
        return decoded



# +

class AutoEncoder(nn.Module):
    def __init__(self, conv=True, input_dim=2880):
        super().__init__()
        if conv:
            print("Using ConvEncoder")
            self.encoder = ConvEncoder()
            self.decoder = ConvDecoder()
        else:
            if input_dim == 2880:
                print("Using LinearEncoder")
                self.encoder = LinearEncoder2880()
                self.decoder = LinearDecoder2880()
            else:
                self.encoder = LinearEncoder24()
                self.decoder = LinearDecoder24()
                
    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded, encoded 
    
class AutoEncoderNet(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, encoded = y_pred  # <- unpack the tuple that was returned by `forward`
        loss_reconstruction = super().get_loss(decoded, y_true, *args, **kwargs)
        loss_l1 = 1e-3 * torch.abs(encoded).sum()
        return loss_reconstruction + loss_l1  



# -

net = AutoEncoderNet(
    AutoEncoder,
    max_epochs=10000,
    lr=0.0000001,
    #criterion=nn.MSELoss(),
    module__conv=False,
    module__input_dim=24,
    iterator_train__batch_size=16,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    #device="cuda",
)

# +
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#X = df.fillna(0.0).values.astype(np.float32)
X = df_hour.fillna(0.0).values.astype(np.float32)

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

net.fit(X, X)

# Problem using the pipeline: It does not scale "Y"
# pipe = Pipeline([
#     ('scale', StandardScaler()),
#     ('net', net),
# ])
# pipe.fit(X, X)

#X = np.expand_dims(X, axis=1)
#X = df_hour.fillna(0.0).values.astype(np.float32)
#y_proba = net.predict_proba(X)
# -

scaler.inverse_transform(net.predict(X)[0])
#net.forward(X)

# +
# PLAYGROUND
input = torch.randn(1, 1, 2880)
inputnet = torch.randn(1, 2880)

c = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=20, stride=10, padding=0)
c2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=6, stride=4, padding=0)
mp = nn.MaxPool1d(kernel_size=2,stride=2)

res = c(input)
print("CNN1:", res.shape)

res = c2(res)
print("CNN2:", res.shape)

#res = c2(c(input.unsqueeze(1))) #.view(-1).shape
#mp(res).shape, res.shape
#res.view(res.size(0), -1).shape

# print(mp(c2(mp(c(input.unsqueeze(1))))).shape)

#c(input.unsqueeze(0))

e = ConvEncoder()
out = e(inputnet)
print(out.shape)

d = ConvDecoder()
d(out).shape

# dc = nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=6, stride=4, padding=0)
# dec1 = dc(out)
# print(dec1.shape)
# dc2 = nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size=10, stride=30, padding=0)
# dc2(dec1).shape

#ae = AutoEncoder()
#ae.fit()

#X.sum(axis=2)

#X.shape

