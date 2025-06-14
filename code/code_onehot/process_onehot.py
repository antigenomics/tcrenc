import numpy as np
import pandas as pd
import argparse
import warnings
from pathlib import Path
import sys

# libs for ml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')
current_file_path = Path(__file__).absolute()
project_root = current_file_path.parent.parent.parent
sys.path.append(str(project_root))

# our module with some func
import modules.modules_onehot.pepcode as pepcode
from modules.modules_onehot.autoencoder import Autoencoder
import modules.modules_onehot.constants as constants
from modules.modules_kidera.gpu import GPU


latent_dims = constants.LATENT_DIMS
batch_size = 400
use_gpu = constants.USE_GPU
loss_function = nn.CrossEntropyLoss()
max_cdr3_len = 19
max_ep_len = 20
device = GPU(use_gpu)


def process(input_file: str, output_dir: str) -> None:
    inp_data = pd.read_csv(input_file)

    # Fetch original lists
    inp_data_cdr3_list_ori = inp_data.cdr3.values
    inp_data_ep_list_ori = inp_data.antigen_epitope.values

    # Gap insertion. Make new list with all gap variants.
    inp_data_cdr3_list = []
    inp_data_ep_list = []
    for cdr3, ep in zip(inp_data_cdr3_list_ori, inp_data_ep_list_ori):
        gap_count_cdr3 = max_cdr3_len - len(cdr3)
        gap_count_ep = max_ep_len - len(ep)

        inp_data_cdr3_list.append(cdr3[0:3]+'-'*gap_count_cdr3+cdr3[3:])
        inp_data_cdr3_list.append(cdr3[0:4]+'-'*gap_count_cdr3+cdr3[4:])
        inp_data_cdr3_list.append(cdr3[0:-3]+'-'*gap_count_cdr3+cdr3[-3:])
        inp_data_cdr3_list.append(cdr3[0:-4]+'-'*gap_count_cdr3+cdr3[-4:])

        inp_data_ep_list.append(ep[0:3]+'-'*gap_count_ep+ep[3:])
        inp_data_ep_list.append(ep[0:4]+'-'*gap_count_ep+ep[4:])
        inp_data_ep_list.append(ep[0:-3]+'-'*gap_count_ep+ep[-3:])
        inp_data_ep_list.append(ep[0:-4]+'-'*gap_count_ep+ep[-4:])

    len_cdr3 = max_cdr3_len
    len_ep = max_ep_len

    # To One-hot matrix
    inp_data_cdr3_oh = np.zeros((len(inp_data_cdr3_list),
                                 len(pepcode.AA_LIST),
                                 len_cdr3),
                                dtype=np.float32)
    for i in range(len(inp_data_cdr3_oh)):
        inp_data_cdr3_oh[i] = pepcode.one_hot_code(inp_data_cdr3_list[i])

    inp_data_ep_oh = np.zeros((len(inp_data_ep_list),
                               len(pepcode.AA_LIST),
                               len_ep),
                              dtype=np.float32)
    for i in range(len(inp_data_ep_oh)):
        inp_data_ep_oh[i] = pepcode.one_hot_code(inp_data_ep_list[i])

    cdr3_oh_matr_size = inp_data_cdr3_oh[0].size
    ep_oh_matr_size = inp_data_ep_oh[0].size

    # Prepare cdr3 dataloader
    inp_data_cdr3_oh_dataset = TensorDataset(torch.tensor(inp_data_cdr3_oh),
                                             torch.tensor(np.ones(inp_data_cdr3_oh.shape[0])))

    inp_data_cdr3_oh_dl = DataLoader(inp_data_cdr3_oh_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)

    # Prepare epitope dataloader
    inp_data_ep_oh_dataset = TensorDataset(torch.tensor(inp_data_ep_oh),
                                           torch.tensor(np.ones(inp_data_ep_oh.shape[0])))
    inp_data_ep_oh_dl = DataLoader(inp_data_ep_oh_dataset,
                                   batch_size=batch_size,
                                   shuffle=False)

    model_cdr3 = Autoencoder(399)
    model_cdr3 = torch.load('./models/models_onehot/cdr3_model_final.pth',
                            weights_only=False, map_location='cpu')
    model_cdr3 = model_cdr3.to(device)

    # Make cdr3 embeddings
    model_cdr3.eval()

    output = []
    loss_avg, num_batches = 0, 0

    for (pep, _) in inp_data_cdr3_oh_dl:
        with torch.no_grad():
            pep_o = pep
            pep_o = pep_o.to(device)
            pep = pep.reshape(-1, cdr3_oh_matr_size)
            pep = pep.to(device)
            pep_encod = model_cdr3.encoding(pep)
            pep_recon = model_cdr3(pep)
            pep_recon_rs = pep_recon.reshape(pep_o.shape)
            loss = loss_function(pep_recon_rs, pep_o)
            loss_avg += loss.item()
            num_batches += 1
        output.append((pep, pep_encod))
    loss_avg /= num_batches
    print('Average reconstruction error of cdr3 sequences on sample: %f' % (loss_avg))

    cdr3_embd = np.zeros((len(inp_data_cdr3_list), latent_dims),
                         dtype=np.float32)
    pointer = 0
    for i in range(num_batches):
        cur_batch_size = len(output[i][0])
        cdr3_embd[pointer:pointer + cur_batch_size, :] = output[i][1].reshape((cur_batch_size, latent_dims)).numpy(force=True)
        pointer += cur_batch_size

    cdr3_embd_rs = cdr3_embd.reshape(len(inp_data_cdr3_list_ori), 4*64)

    model_ep = Autoencoder(420)
    model_ep = torch.load('./models/models_onehot/epitope_model_final.pth',
                          weights_only=False, map_location='cpu')
    model_ep = model_ep.to(device)

    # Make epitope embeddings
    model_ep.eval()

    output = []
    loss_avg, num_batches = 0, 0

    for (pep, _) in inp_data_ep_oh_dl:
        with torch.no_grad():
            pep_o = pep
            pep_o = pep_o.to(device)
            pep = pep.reshape(-1, ep_oh_matr_size)
            pep = pep.to(device)
            pep_encod = model_ep.encoding(pep)
            pep_recon = model_ep(pep)
            pep_recon_rs = pep_recon.reshape(pep_o.shape)
            loss = loss_function(pep_recon_rs, pep_o)
            loss_avg += loss.item()
            num_batches += 1
        output.append((pep, pep_encod))
    loss_avg /= num_batches
    print('Average reconstruction error on sample: %f' % (loss_avg))

    ep_embd = np.zeros((len(inp_data_ep_list), latent_dims), dtype=np.float32)
    pointer = 0
    for i in range(num_batches):
        cur_batch_size = len(output[i][0])
        ep_embd[pointer:pointer + cur_batch_size, :] = output[i][1].reshape((cur_batch_size, latent_dims)).numpy(force=True)
        pointer += cur_batch_size

    ep_embd_rs = ep_embd.reshape(len(inp_data_ep_list_ori), 4*64)

    # Saving in csv
    encoded_cdr3 = pd.DataFrame(cdr3_embd_rs)
    encoded_cdr3.to_csv(f'{output_dir}/embeddings_cdr3_onehot.csv',
                        index=False)
    encoded_ep = pd.DataFrame(ep_embd_rs)
    encoded_ep.to_csv(f'{output_dir}/embeddings_epitopes_onehot.csv',
                      index=False)
    print("All files saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    process(args.input, args.output)
