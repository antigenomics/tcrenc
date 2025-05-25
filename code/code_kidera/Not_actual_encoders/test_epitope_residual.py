from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
from autoencoder_residual import ConvAutoEncoderRes
import numpy as np
use_gpu=True
if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
elif use_gpu and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

batch_size = 64
linear = 19  
latent_dim = 64

model=ConvAutoEncoderRes(linear,latent_dim=64)
model.load_state_dict(torch.load('/projects/tcr_nlp/conv_autoencoder/conv_res_block/epitope_res.pth'))

epitope_tensor=torch.load('../../../epitopes_test_tensor.pt')

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


encoded_epitope = pd.DataFrame(encoded_epitope)
encoded_epitope.to_csv('../../../encoded_epitope_kidera_residual.csv', index=False, header=False)