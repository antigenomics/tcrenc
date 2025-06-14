import torch
from autoencoder import ConvAutoEncoder
from modules.modules_kidera.gpu import GPU
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

use_gpu = True
DEVICE = GPU(use_gpu)

BATCH_SIZE = 64
LINEAR = 20
LATENT_DIM = 64

epitope_tensor = torch.load("../../../epitopes_test_tensor.pt")


model = ConvAutoEncoder(linear = LINEAR, latent_dim = LATENT_DIM)
model.load_state_dict(torch.load("/projects/tcr_nlp/conv_autoencoder/conv/epitope.pth"))


def get_encoded_epitope(epitope_enc, model, batch_size, device = DEVICE):
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
    test_loader = DataLoader(TensorDataset(epitope_enc), batch_size = batch_size)
    encoded_epitope, decoded_epitope = [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            latent = model.linear_encode(model.encoder(x))
            decoded = model(x)
            encoded_epitope.append(latent.cpu())
            decoded_epitope.append(decoded.cpu())

    return torch.cat(encoded_epitope), torch.cat(decoded_epitope)


encoded_epitope, decoded_epitope = get_encoded_epitope(epitope_tensor, model, BATCH_SIZE, DEVICE)

encoded_epitope = pd.DataFrame(encoded_epitope)
encoded_epitope.to_csv(
    "../../../encoded_epitope_kidera_residual.csv", index = False, header = False
)
