# Importing
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import argparse
import warnings
warnings.filterwarnings('ignore')
current_file_path = Path(__file__).absolute()

project_root = current_file_path.parent.parent.parent

sys.path.append(str(project_root))
from modules.modules_kidera.kidera import kidera_final_dict
from actual_encoders.autoencoder import ConvAutoEncoder
from Not_actual_encoders.autoencoder_residual import ConvAutoEncoderRes
import torch



# GPU
use_gpu=True
if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
elif use_gpu and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
#Parameters

batch_size = 64
    
# Functions for preprocessing
def sequence_to_factor(sequence, kidera_dict):
    return np.array([kidera_dict[aa] for aa in sequence], dtype=np.float32).T
def insert_x(x: str):
    length_X = 19 - len(x)
    if len(x) == 4:
        return x[:2] + 'X' * 15 + x[2:]
    elif len(x) == 5:
        return x[:2] + 'X' * 7 + x[2] + 'X' * 7 + x[3:]
    elif len(x) == 6:
        return x[:3] + 'X' * 13 + x[3:]
    else:
        pref, suff = x[:3], x[-3:]
        mid = x[3:-3]
        return pref + 'X' * (length_X // 2 + length_X % 2) + mid + 'X' * (length_X // 2) + suff
    
def func_antigen(antigen: str):
    n=20
    start_end=(n-len(antigen))//2
    if len(antigen)%2==0:
        return start_end*'X'+antigen+start_end*'X'
    else:
        return start_end*'X'+antigen[:len(antigen)//2]+'X'+antigen[len(antigen)//2:]+start_end*'X' 
    
def epitope_to_kidera(epitopes: pd.Series,kidera_dict: pd.DataFrame):
    epitopes = epitopes.apply(func_antigen)
    factors_array = np.stack(epitopes.map(lambda seq: sequence_to_factor(seq, kidera_dict)).values, axis=0)
    factors_tensor = torch.tensor(factors_array, dtype=torch.float32).unsqueeze(1) 
    return factors_tensor
def process(input_file: str, output_dir: str, residual_block: bool)->None:
    # Loading and preprocessing
    data=pd.read_csv(input_file)

    data['cdr3'] = data['cdr3'].apply(insert_x)

    data_cdr3=data[['cdr3']]

    epitopes=data[['antigen_epitope']]

    data_train_test_cdr3 = torch.tensor(np.stack(data_cdr3['cdr3'].map(
        lambda seq: sequence_to_factor(seq, kidera_final_dict)
    ).values, axis=0),dtype=torch.float32).unsqueeze(1)

    epitope_tensor=epitope_to_kidera(epitopes['antigen_epitope'],kidera_final_dict)


    if residual_block:
    # Testing
        model = ConvAutoEncoder(linear=19,latent_dim=64).to(device)
        model.load_state_dict(torch.load('/projects/tcr_nlp/conv_autoencoder/conv/cdr3.pth'))
        def get_encoded_cdr(cdr_enc, model, batch_size, device='cuda'):      
            """
            Pass cdr3 encodings through an autoencoder and return both encoded (latent) and decoded outputs.

            Args:
                epitope_enc (Tensor): Input tensor of encoded epitopes (e.g., physicochemical features).
                model (nn.Module): Trained autoencoder model.
                batch_size (int): Batch size for processing.
                device (str): Device for computation ('cuda' or 'cpu').

            Returns:
                Tuple[Tensor, Tensor]: 
                    - Encoded latent representations (shape: [N, latent_dim])
                    - Reconstructed epitopes (same shape as input)
            """
            model.eval()
            model.to(device)
            test_loader = DataLoader(TensorDataset(cdr_enc), batch_size=batch_size)
            encoded_cdr3, decoded_cdr3 = [], []

            with torch.no_grad():
                for batch in test_loader:
                    x = batch[0].to(device)
                    latent = model.linear_encode(model.encoder(x))
                    decoded = model(x)
                    encoded_cdr3.append(latent.cpu())
                    decoded_cdr3.append(decoded.cpu())
            return torch.cat(encoded_cdr3), torch.cat(decoded_cdr3)

        encoded_cdr3,decoded_cdr3 = get_encoded_cdr(data_train_test_cdr3,model,batch_size)

        #Saving
        encoded_cdr3 = pd.DataFrame(encoded_cdr3)
        encoded_cdr3.to_csv(f'{output_dir}/encoded_cdr3_kidera.csv', index=False, header=False)



        # Testing
        model=ConvAutoEncoder(linear=20,latent_dim=64)
        model.load_state_dict(torch.load('/projects/tcr_nlp/conv_autoencoder/conv/epitope.pth'))
        def get_encoded_epitope(epitope_enc, model, batch_size, device='cuda'):        
            """
            Pass epitope encodings through an autoencoder and return both encoded (latent) and decoded outputs.

            Args:
                epitope_enc (Tensor): Input tensor of encoded epitopes (e.g., physicochemical features).
                model (nn.Module): Trained autoencoder model.
                batch_size (int): Batch size for processing.
                device (str): Device for computation ('cuda' or 'cpu').

            Returns:
                Tuple[Tensor, Tensor]: 
                    - Encoded latent representations (shape: [N, latent_dim])
                    - Reconstructed epitopes (same shape as input)
            """
            model.eval()
            model.to(device)
            test_loader = DataLoader(TensorDataset(epitope_enc), batch_size=batch_size)
            encoded_epitope, decoded_epitope = [], []

            with torch.no_grad():
                for batch in test_loader:
                    x = batch[0].to(device)
                    latent = model.linear_encode(model.encoder(x))
                    decoded = model(x)
                    encoded_epitope.append(latent.cpu())
                    decoded_epitope.append(decoded.cpu())

            return torch.cat(encoded_epitope), torch.cat(decoded_epitope)
        encoded_epitope,decoded_epitope = get_encoded_epitope(epitope_tensor,model,64)

        #Saving
        encoded_epitope = pd.DataFrame(encoded_epitope)
        encoded_epitope.to_csv(f'{output_dir}/encoded_epitope_kidera.csv', index=False, header=False)


    else:
        #Testing

        model = ConvAutoEncoderRes(linear=19,latent_dim=64).to(device)
        model.load_state_dict(torch.load('/projects/tcr_nlp/conv_autoencoder/conv_res_block/cdr3_res.pth'))
        def get_encoded_cdr(cdr_enc, model, batch_size, device='cuda'):    
            """
            Pass epitope encodings through an autoencoder and return both encoded (latent) and decoded outputs.

            Args:
                epitope_enc (Tensor): Input tensor of encoded epitopes (e.g., physicochemical features).
                model (nn.Module): Trained autoencoder model.
                batch_size (int): Batch size for processing.
                device (str): Device for computation ('cuda' or 'cpu').

            Returns:
                Tuple[Tensor, Tensor]: 
                    - Encoded latent representations (shape: [N, latent_dim])
                    - Reconstructed epitopes (same shape as input)
            """
            model.eval()
            model.to(device)
            test_loader = DataLoader(TensorDataset(cdr_enc), batch_size=batch_size)
            encoded_cdr3, decoded_cdr3 = [], []

            with torch.no_grad():
                for batch in test_loader:
                    x = batch[0].to(device)
                    latent = model.linear_encode(model.encoder(x))
                    decoded = model(x)
                    encoded_cdr3.append(latent.cpu())
                    decoded_cdr3.append(decoded.cpu())
            return torch.cat(encoded_cdr3), torch.cat(decoded_cdr3)

        encoded_cdr3,decoded_cdr3 = get_encoded_cdr(data_train_test_cdr3,model,batch_size)

        #Saving
        encoded_cdr3 = pd.DataFrame(encoded_cdr3)
        encoded_cdr3.to_csv(f'{output_dir}/encoded_cdr3_kidera_residual.csv', index=False, header=False)


        #Testing
        model=ConvAutoEncoderRes(linear=20,latent_dim=64)
        model.load_state_dict(torch.load('/projects/tcr_nlp/conv_autoencoder/conv_res_block/epitope_res.pth'))


        def get_encoded_epitope(epitope_enc, model, batch_size, device='cuda'):        
            """
            Pass epitope encodings through an autoencoder and return both encoded (latent) and decoded outputs.

            Args:
                epitope_enc (Tensor): Input tensor of encoded epitopes (e.g., physicochemical features).
                model (nn.Module): Trained autoencoder model.
                batch_size (int): Batch size for processing.
                device (str): Device for computation ('cuda' or 'cpu').

            Returns:
                Tuple[Tensor, Tensor]: 
                    - Encoded latent representations (shape: [N, latent_dim])
                    - Reconstructed epitopes (same shape as input)
            """
            model.eval()
            model.to(device)
            test_loader = DataLoader(TensorDataset(epitope_enc), batch_size=batch_size)
            encoded_epitope, decoded_epitope = [], []

            with torch.no_grad():
                for batch in test_loader:
                    x = batch[0].to(device)
                    latent = model.linear_encode(model.encoder(x))
                    decoded = model(x)
                    encoded_epitope.append(latent.cpu())
                    decoded_epitope.append(decoded.cpu())

            return torch.cat(encoded_epitope), torch.cat(decoded_epitope)
        encoded_epitope,decoded_epitope = get_encoded_epitope(epitope_tensor,model,64)

        #Saving
        encoded_epitope = pd.DataFrame(encoded_epitope)
        encoded_epitope.to_csv(f'{output_dir}/encoded_epitope_kidera_residual.csv', index=False, header=False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--residual_block', type=str, required=True)
    
    args = parser.parse_args()
    process(args.input, args.output, args.residual_block.lower() == 'true')
