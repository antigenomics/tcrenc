import numpy as np
import pandas as pd
from datetime import date
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
import datetime
from collections import Counter
import argparse


# libs for ml
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler, robust_scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

# my module with some func
import pepcode
import warnings
warnings.filterwarnings('ignore')

AA_LIST = pepcode.AA_LIST

latent_dims = 64
batch_size = 400
learning_rate = 1e-4 
use_gpu = True

# Device set
if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
elif use_gpu and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

    
    
    

# Autoencoder's defining
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self,in_out,latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_out, out_features=latent_dims),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dims, out_features=in_out),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encoding(self, x):
        encoded = self.encoder(x)
        return encoded

    def decoding(self, encoded):
        decoded = self.decoder(encoded)
        return decoded
    
    
    
def process(input_file: str,output_dir: str)->None: 
    X_test = pd.read_csv(input_file)


    # To lists
    X_test_cdr3_list_ori = X_test.cdr3.values
    X_test_ep_list_ori = X_test.antigen_epitope.values

    X_test_cdr3_list_ori = X_test.cdr3.values
    X_test_ep_list_ori = X_test.antigen_epitope.values

    #Gap insertion (test)

    X_test_cdr3_list = []
    X_test_epitops_list = []
    for cdr3, ep in zip(X_test_cdr3_list_ori, X_test_ep_list_ori):
        gap_count_for_cdr3 = 19-len(cdr3) # max_len = 19
        gap_count_for_ep = 20-len(ep) # max_len = 20

        X_test_cdr3_list.append(cdr3[0:3]+'-'*gap_count_for_cdr3+cdr3[3:])
        X_test_cdr3_list.append(cdr3[0:4]+'-'*gap_count_for_cdr3+cdr3[4:])
        X_test_cdr3_list.append(cdr3[0:-3]+'-'*gap_count_for_cdr3+cdr3[-3:])
        X_test_cdr3_list.append(cdr3[0:-4]+'-'*gap_count_for_cdr3+cdr3[-4:])

        X_test_epitops_list.append(ep[0:3]+'-'*gap_count_for_ep+ep[3:])
        X_test_epitops_list.append(ep[0:4]+'-'*gap_count_for_ep+ep[4:])
        X_test_epitops_list.append(ep[0:-3]+'-'*gap_count_for_ep+ep[-3:])
        X_test_epitops_list.append(ep[0:-4]+'-'*gap_count_for_ep+ep[-4:])

    len_cdr3 = len(X_test_cdr3_list[0])
    len_ep = len(X_test_epitops_list[0])

    #To One-hot (Test)
    X_test_cdr3_oh = np.zeros((len(X_test_cdr3_list), len(pepcode.AA_LIST), len_cdr3), dtype = np.float32)
    for i in range(len(X_test_cdr3_oh)):
        X_test_cdr3_oh[i] = pepcode.one_hot_code(X_test_cdr3_list[i])

    X_test_ep_oh = np.zeros((len(X_test_epitops_list), len(pepcode.AA_LIST), len_ep), dtype = np.float32)
    for i in range(len(X_test_ep_oh)):
        X_test_ep_oh[i] = pepcode.one_hot_code(X_test_epitops_list[i])

    X_test_cdr3_oh = np.zeros((len(X_test_cdr3_list), len(pepcode.AA_LIST), len_cdr3), dtype = np.float32)
    for i in range(len(X_test_cdr3_oh)):
        X_test_cdr3_oh[i] = pepcode.one_hot_code(X_test_cdr3_list[i])

    X_test_ep_oh = np.zeros((len(X_test_epitops_list), len(pepcode.AA_LIST), len_ep), dtype = np.float32)
    for i in range(len(X_test_ep_oh)):
        X_test_ep_oh[i] = pepcode.one_hot_code(X_test_epitops_list[i])

    # Prepare cdr3 dataloader (test)
    X_test_cdr3_oh_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_cdr3_oh), torch.tensor(np.ones(X_test_cdr3_oh.shape[0])))
    X_test_cdr3_oh_dl = torch.utils.data.DataLoader(X_test_cdr3_oh_dataset, batch_size=batch_size, shuffle=False)

    # Prepare ep dataloader (test)
    X_test_ep_oh_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_ep_oh), torch.tensor(np.ones(X_test_ep_oh.shape[0])))
    X_test_ep_oh_dl = torch.utils.data.DataLoader(X_test_ep_oh_dataset, batch_size=batch_size, shuffle=False)

    cdr3_oh_matr_size = X_test_cdr3_oh[0].size
    ep_oh_matr_size = X_test_ep_oh[0].size


    model_cdr3 = Autoencoder(399,64)
    model_cdr3 = torch.load('./models/models_onehot/cdr3_model_final.pth', weights_only=False,map_location='cpu')

    model_cdr3 = model_cdr3.to(device)
    loss_function = nn.CrossEntropyLoss()

    # Make X_train embeddings
    model_cdr3.eval()

    output = [] # For hiddens
    test_loss_avg, num_batches = 0, 0

    for (pep, _) in X_test_cdr3_oh_dl:
        with torch.no_grad():
            pep_o = pep
            pep_o = pep_o.to(device)
            pep = pep.reshape(-1, cdr3_oh_matr_size)
            pep = pep.to(device)
            pep_encod = model_cdr3.encoding(pep)
            pep_recon = model_cdr3(pep)
            pep_recon_rs = pep_recon.reshape(pep_o.shape)
            loss = loss_function(pep_recon_rs, pep_o)
            test_loss_avg += loss.item()
            num_batches += 1
        output.append((pep, pep_encod))
    test_loss_avg /= num_batches
    print('Average reconstruction error on sample: %f' % (test_loss_avg))

    X_test_cdr3_embd = np.zeros((len(X_test_cdr3_list), latent_dims), dtype = np.float32)


    pointer = 0
    for i in range(num_batches):
        cur_batch_size = len(output[i][0])
        X_test_cdr3_embd[pointer:pointer + cur_batch_size, :] = output[i][1].reshape((cur_batch_size, latent_dims)).numpy(force=True)
        pointer += cur_batch_size

    X_test_cdr3_embd_rs = X_test_cdr3_embd.reshape(len(X_test_cdr3_list_ori), 4*64)

    model_ep = Autoencoder(420,64)
    model_ep= torch.load('./models/models_onehot/epitope_model_final.pth', weights_only=False,map_location='cpu')

    model_ep = model_ep.to(device)
    loss_function = nn.CrossEntropyLoss()

    # Make epitope embeddings
    model_ep.eval()

    output = [] # For hiddens
    test_loss_avg, num_batches = 0, 0

    for (pep, _) in X_test_ep_oh_dl:
        with torch.no_grad():
            pep_o = pep
            pep_o = pep_o.to(device)
            pep = pep.reshape(-1, ep_oh_matr_size)
            pep = pep.to(device)
            pep_encod = model_ep.encoding(pep)
            pep_recon = model_ep(pep)
            pep_recon_rs = pep_recon.reshape(pep_o.shape)
            loss = loss_function(pep_recon_rs, pep_o)
            test_loss_avg += loss.item()
            num_batches += 1
        output.append((pep, pep_encod))
    test_loss_avg /= num_batches
    print('Average reconstruction error on sample: %f' % (test_loss_avg))


    X_test_ep_embd = np.zeros((len(X_test_epitops_list), latent_dims), dtype = np.float32)

    pointer = 0
    for i in range(num_batches):
        cur_batch_size = len(output[i][0])
        X_test_ep_embd[pointer:pointer + cur_batch_size, :] = output[i][1].reshape((cur_batch_size, latent_dims)).numpy(force=True)
        pointer += cur_batch_size

    X_test_ep_embd_rs = X_test_ep_embd.reshape(len(X_test_ep_list_ori), 4*64)

    split_cdr3 = np.split(X_test_cdr3_embd_rs, indices_or_sections=4, axis=1)
    names=['3','4','-3','-4']
    # Saving in CSV
    for i, part in enumerate(split_cdr3, start=0):
        df = pd.DataFrame(part)
        df.to_csv(f'{output_dir}/cdr3_embbed_{names[i]}.csv', index=False, header=False)

    split_epitope = np.split(X_test_ep_embd_rs, indices_or_sections=4, axis=1)
    names=['3','4','-3','-4']
    # Saving in CSV
    for i, part in enumerate(split_epitope, start=0):
        df = pd.DataFrame(part)
        df.to_csv(f'{output_dir}/epitope_embbed_{names[i]}.csv', index=False, header=False)

    print("All files saved!")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    process(args.input, args.output)
