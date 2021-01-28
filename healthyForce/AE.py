#!/usr/bin/env python
# coding: utf-8

# Links:
# - [imports](#imports)
# - [Pytorch Lightning](#pytorch_lightning)
# - [Train](#train)
# - [Plot](#plot)

# # Imports <a id='imports'></a>

# In[39]:


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
from sklearn.preprocessing import StandardScaler

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


# In[40]:


import matplotlib.pyplot as plt

def my_plot(epochs, loss):
    plt.plot(epochs, loss)
    
def train(num_epochs,optimizer,criterion,model):
    loss_vals=  []
    for epoch in range(num_epochs):
        epoch_loss= []
        for i, (images, labels) in enumerate(trainloader):
            # rest of the code
            loss.backward()
            epoch_loss.append(loss.item())
            # rest of the code
        # rest of the code
        loss_vals.append(sum(epoch_loss)/len(epoch_loss))
        # rest of the code
    
    # plotting
    my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), loss_vals)

my_plot([1, 2, 3, 4, 5], [100, 90, 60, 30, 10])


# In[41]:



df = pd.read_csv("acm_health_sleep_data-main/processed_hchs/HCHS_per_hour.csv",  converters={"raw_pa": lambda x: np.fromstring(x, sep=',')})
#df = pd.read_csv("mesa/MESA_per_hour.csv",  converters={"raw_pa": lambda x: np.fromstring(x, sep=',')})

rawpa = []
for m in range(0, 60):
    for s in (0, 30):
        rawpa.append("hyp_act_x_%s_%s" % (str(m).zfill(2), str(s).zfill(2)))

df = df[["pid", "ml_sequence", "hyp_time_col", *rawpa]].fillna(0.0).drop_duplicates()

df = df.pivot(["pid", "ml_sequence"], columns=["hyp_time_col"])
df.columns = df.columns.swaplevel(0, 1)
df.sort_index(axis=1, level=0, inplace=True)


### DF hours
dfs = []
for hour in range(0,24):
    dfs.append(df[hour].mean(axis=1))

df_hour = pd.concat(dfs, axis=1)


# -


# In[42]:


df_hour


# In[43]:



net = AutoEncoderNet(
    AutoEncoder,
    max_epochs=20,
    lr=0.0000001,
    #criterion=nn.MSELoss(),
    module__conv=False,
    module__input_dim=24,
    iterator_train__batch_size=16,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    #device="cuda",
)

#X = df.fillna(0.0).values.astype(np.float32)
df_hour = df_hour.fillna(0.0)
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


# # Tests

# In[5]:


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


# # Pytorch Lightning <a id='pytorch_lightning'></a>

# ## Encoder/Decoder definition

# In[44]:


import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, dataset
from pytorch_lightning.loggers import TensorBoardLogger    
from sklearn.preprocessing import OneHotEncoder

import pytorch_lightning as pl


class AESkipConnection(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(2880, 1440)
        self.linear2 = nn.Linear(1440, 360)
        self.linear3 = nn.Linear(360, 90)
        self.linear4 = nn.Linear(90, 30)
        
        self.linear5 = nn.Linear(30, 90)
        self.linear6 = nn.Linear(90, 360)
        self.linear7 = nn.Linear(360, 1440)
        self.linear8 = nn.Linear(1440, 2880)

        
    def forward(self, X):
        ### Encoder
        l1_out = self.linear1(X)
        out = F.relu(l1_out)
        
        l2_out = self.linear2(out)
        out = F.relu(l2_out)
        
        l3_out = self.linear3(out)
        out = F.relu(l3_out)
        
        l4_out = self.linear4(out)
        out = F.relu(l4_out)
        
        ### Decoder
        out = self.linear5(out)
        out += l3_out
        out = F.relu(out)
        
        # out = torch.cat((out, l1_out), 1)
        out = self.linear6(out)
        out += l2_out
        out = F.relu(out)
        
        # out = torch.cat((out, l2_out), 1)
        out = self.linear7(out)
        out += l1_out
        out = F.relu(out)
        
        out = self.linear8(out)
        out = F.relu(out)

        return out
    

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

class LinearEncoder120(nn.Module):
    def __init__(self, label_dim=24):
        super().__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(120+label_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
            
    def forward(self, X):
        encoded = self.encode(X)
        return encoded
    
class LinearDecoder120(nn.Module):
    def __init__(self, label_dim=24):
        super().__init__()
        
        self.decode = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 120+label_dim), nn.ReLU(),
        )

    def forward(self, X):
        decoded = self.decode(X)
        return decoded

class LinearEncoder24(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(24, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4)
        )
                
    def forward(self, X):
        encoded = self.encode(X)
        return encoded

    
class LinearDecoder24(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decode = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 24)
        )

    def forward(self, X):
        decoded = self.decode(X)
        return decoded
    


# ## AE Definition

# In[45]:



    
class LitAutoEncoder(pl.LightningModule):

    def __init__(self, input_dim=24):
        super().__init__()
        
        self.skipconnection = AESkipConnection()
        if(input_dim == 2880):
            self.encoder = LinearEncoder2880()
            self.decoder = LinearDecoder2880()
        else:
            self.encoder = LinearEncoder24()
            self.decoder = LinearDecoder24()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
    
class ResidualAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.skipconnection = AESkipConnection()
    
    def forward(self, x):
        return self.skipconnection(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.skipconnection(x)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.skipconnection(x)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
    

    


# ## VAE Definition

# In[46]:



class VAE(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    Example::
        # not pretrained
        vae = VAE()
    """
    def __init__(
        self,
        enc_out_dim: int = 4,
        kl_coeff: float = 0.1,
        latent_dim: int = 4,
        lr: float = 1e-2,
        input_dim: int = 24,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        if(self.input_dim == 2880):
            self.encoder = LinearEncoder2880()
            self.decoder = LinearDecoder2880()
        elif(self.input_dim == 120):
            # CVAE
            self.encoder = LinearEncoder120()
            self.decoder = LinearDecoder120()
        else:
            self.encoder = LinearEncoder24()
            self.decoder = LinearDecoder24()

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)        
    
    def forward(self, x):
        x = self.encoder(x)
        # mu = self.fc_mu(x)
        # log_var = self.fc_var(x)
        # p, q, z = self.sample(mu, log_var)
        # return self.decoder(z)
        return x

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# In[47]:


train_ids = pd.read_csv('acm_health_sleep_data-main/processed_hchs/HCHS_pid_train.csv')
test_ids = pd.read_csv('acm_health_sleep_data-main/processed_hchs/HCHS_pid_test.csv')

class AEDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.scaler = preprocessing.StandardScaler()
        self.df = df.fillna(0.0).values.astype(np.float32)
        
        self.scaler.fit(self.df)
        self.df = self.scaler.transform(self.df)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Select sample
        X = self.df[index]
        y = self.df[index]

        return X, y
    
    def unscale(self, x):
        return self.scaler.inverse_transform(x)

class CVAEDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.scaler = preprocessing.StandardScaler()
        self.df = df.fillna(0.0)
        
        hour_per_sample = self.df.index.get_level_values(2)
        hour_per_sample = hour_per_sample.values.reshape(-1, 1)
        
        # One-Hot-Encode the labels
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        self.label = enc.fit_transform(hour_per_sample)
        self.label = self.label.astype(np.float32)
        self.categories_ = enc.categories_
        
        # Scale
        self.df = self.df.values.astype(np.float32)
        
        self.scaler.fit(self.df)
        self.df = self.scaler.transform(self.df)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Select sample
        if(type(index) == int):
            # concatenation on axis 0 since we're getting one item
            X = np.concatenate((self.df[index], self.label[index]), axis=0)
            y = np.concatenate((self.df[index], self.label[index]), axis=0)
        else:
            X = np.concatenate((self.df[index], self.label[index]), axis=1)
            y = np.concatenate((self.df[index], self.label[index]), axis=1)

        return X, y
    
    def unscale(self, x):
        return self.scaler.inverse_transform(x)

    
D_train_2880 = AEDataset(df.loc[train_ids.pid])    
D_test_2880 = AEDataset(df.loc[test_ids.pid])    

D_train_24 = AEDataset(df_hour.loc[train_ids.pid])
D_test_24 = AEDataset(df_hour.loc[test_ids.pid])

train_loader_2880 = DataLoader(D_train_2880, batch_size=512)
test_loader_2880 = DataLoader(D_test_2880, batch_size=512)

train_loader_24 = DataLoader(D_train_24, batch_size=512)
test_loader_24 = DataLoader(D_test_24, batch_size=512)

# Form CVAE input
df_cvae = []
for i in range(24):
    df_cvae.append(df[i])

## Make the index (pid, sequence, hour of day)
df_cvae = pd.concat(df_cvae, keys=range(24), names=['hour'])
df_cvae = df_cvae.swaplevel(0, 1)
df_cvae = df_cvae.swaplevel(1, 2)

cvae_train = CVAEDataset(df_cvae.loc[train_ids.pid])
cvae_test = CVAEDataset(df_cvae.loc[test_ids.pid])

cvae_train_loader = DataLoader(cvae_train, batch_size=512)
cvae_test_loader = DataLoader(cvae_test, batch_size=512)


# # Train AE <a id='train'></a>

# In[51]:


from pytorch_lightning.loggers import CSVLogger

def train_model(model, train, test, exp_name, epochs=100):  
    pl.seed_everything(42)
    logger = CSVLogger("logs", name=exp_name)
    
    trainer = pl.Trainer(gpus=1, max_epochs=epochs, deterministic=True, logger=logger)
    trainer.fit(model, train, test)
    return trainer


# ## AE2880

# In[52]:


# init model
autoencoder_2880 = ResidualAutoEncoder()
train_model(autoencoder_2880, train_loader_2880, test_loader_2880, 'ae2880', epochs=30)


# In[123]:


test = torch.Tensor(D_train_2880.df)


# In[131]:


autoencoder_2880(test[:1])


# In[142]:


plot_train_test_loss('ae2880')


# ## AE24

# In[50]:


# init model
autoencoder_24 = LitAutoEncoder(input_dim=24)
train_model(autoencoder_24, train_loader_24, test_loader_24, 'ae24')


# # Train VAE

# ## VAE2880

# In[53]:


# init model
vae_2880 = VAE(input_dim=2880, enc_out_dim=30, latent_dim=30)
train_model(vae_2880, train_loader_2880, test_loader_2880, 'vae2880')


# ## VAE24

# In[54]:


# init model
vae_24 = VAE(input_dim=24)
train_model(vae_24, train_loader_24, test_loader_24, 'vae24')


# # Train CVAE

# In[55]:


df_cvae


# In[56]:


# init model
cvae = VAE(input_dim=120, enc_out_dim=8, latent_dim=8)
train_model(cvae, cvae_train_loader, cvae_test_loader, 'cvae')


# # Plot <a id='plot'></a>

# In[57]:


import matplotlib.pyplot as plt
import os
import re

def plot_train_test_loss(m):
    fig, ax = plt.subplots(1)
    
    log_file = load_logs(m)
    
    ax.plot(log_file['train_loss'])
    ax.plot(log_file['val_loss'])
    ax.set_title(m + ' model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper right')
    plt.savefig(m)

def load_logs(model):
    # pl loggers logs validation, then training in two separate rows
    # so I join the two rows together via .first()
    
    # get the last version
    last_version = os.listdir('logs/' + model)
    last_version.sort(key=natural_keys)
    last_version = last_version[-1]
    
    log_file = pd.read_csv('logs/' + model + '/' + last_version + '/metrics.csv')
    log_file = log_file.groupby('epoch').first()
    return log_file

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



models = ['ae2880', 'ae24', 'vae2880', 'vae24', 'cvae']

for i, m in enumerate(models):
    plot_train_test_loss(m)


# # Generate embeddings

# In[60]:


train = torch.Tensor(D_train_2880.df)
test = torch.Tensor(D_test_2880.df)

AE_embedded_train_2880 = pd.DataFrame(autoencoder_2880(train).detach().numpy())
AE_embedded_test_2880 = pd.DataFrame(autoencoder_2880(test).detach().numpy())
VAE_embedded_train_2880 = pd.DataFrame(vae_2880.forward(train).detach().numpy())
VAE_embedded_test_2880 = pd.DataFrame(vae_2880.forward(test).detach().numpy())

train = torch.Tensor(D_train_24.df)
test = torch.Tensor(D_test_24.df)

AE_embedded_train_24 = pd.DataFrame(autoencoder_24(train).detach().numpy())
AE_embedded_test_24 = pd.DataFrame(autoencoder_24(test).detach().numpy())
VAE_embedded_train_24 = pd.DataFrame(vae_24(train).detach().numpy())
VAE_embedded_test_24 = pd.DataFrame(vae_24(test).detach().numpy())

train = torch.Tensor(cvae_train[:][0])
test = torch.Tensor(cvae_test[:][0])

cvae_embedded_train = pd.DataFrame(cvae(train).detach().numpy())
cvae_embedded_test = pd.DataFrame(cvae(test).detach().numpy())

train = torch.Tensor(cvae_train[:][0])
test = torch.Tensor(cvae_test[:][0])

# (219936 / 9164) = 24
# 8*24
features_for_24_hours = cvae_embedded_train.shape[1]*(cvae_embedded_train.shape[0] // D_train_2880.df.shape[0])

cvae_embedded_train = cvae(train).detach().numpy()
cvae_embedded_train = cvae_embedded_train.reshape(D_train_2880.df.shape[0], features_for_24_hours)
cvae_embedded_train = pd.DataFrame(cvae_embedded_train)

cvae_embedded_test = cvae(test).detach().numpy()
cvae_embedded_test = cvae_embedded_test.reshape(D_test_2880.df.shape[0], features_for_24_hours)
cvae_embedded_test = pd.DataFrame(cvae_embedded_test)


# In[61]:


embeddings_train = [AE_embedded_train_2880, 
                    AE_embedded_train_24, 
                    VAE_embedded_train_2880, 
                    VAE_embedded_train_24,
                    cvae_embedded_test]

embeddings_train = pd.concat(embeddings_train, 
                               keys=['ae2880', 'ae24', 'vae2880', 'vae24', 'cvae'], 
                               names=['joined_dfs'], axis=1)

embeddings_train.index = df_hour.loc[train_ids.pid].index

embeddings_test = [AE_embedded_test_2880, 
                   AE_embedded_test_24, 
                   VAE_embedded_test_2880, 
                   VAE_embedded_test_24,
                   cvae_embedded_test]

embeddings_test = pd.concat(embeddings_test, 
                               keys=['ae2880', 'ae24', 'vae2880', 'vae24', 'cvae'], 
                               names=['joined_dfs'], axis=1)
embeddings_test.index = df_hour.loc[test_ids.pid].index


# In[77]:


embeddings_train


# In[78]:


# remove joined_df column name
embeddings_train.columns = embeddings_train.columns.rename(names=[None, None])
# embeddings_train.to_csv('embeddings_train.csv', index=True, index_label=False)
embeddings_train.to_pickle('embeddings_train.pkl')

embeddings_test.columns = embeddings_test.columns.rename(names=[None, None])
# embeddings_test.to_csv('embeddings_test.csv', index=True, index_label=False)
embeddings_test.to_pickle('embeddings_test.pkl')


# In[143]:


t = torch.tensor(X[0])
z = autoencoder(t)
x_hat = autoencoder.decoder(z)
print(x_hat)


# In[ ]:


# 11 folds, 1 test set
# Different Representation 
# - AE Raw 2880 (will be bad)
# - AE 24-hour (will be better)
# - VAE (generate data per hour)
# - Conditional per hour or per day

# Append it to X Different features ['ae2880', 'ae24', '...']
# The embedded features (30 + 8)

# Train cvae
# plot
# join embeddings
# send
# commit

