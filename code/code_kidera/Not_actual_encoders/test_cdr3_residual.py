from modules.modules_kidera.gpu import GPU
import torch
from autoencoder_residual import ConvAutoEncoderRes
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


USE_GPU = True
BATCH_SIZE = 64
LINEAR = 19
LATENT_DIM = 64

device = GPU(USE_GPU)

data_train_cdr3 = torch.load("../../../data_train_test_cdr3.pt")

model = ConvAutoEncoderRes(linear=LINEAR, latent_dim=LATENT_DIM).to(device)
model.load_state_dict(
    torch.load("/projects/tcr_nlp/conv_autoencoder/conv_res_block/cdr3_res.pth")
)


def get_encoded_cdr(cdr_enc, model, batch_size, device="cuda"):
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


encoded_cdr3, decoded_cdr3 = get_encoded_cdr(data_train_cdr3, model, BATCH_SIZE)

encoded_cdr3 = pd.DataFrame(encoded_cdr3)
encoded_cdr3.to_csv(
    "../../../encoded_cdr3_kidera_residual.csv", index=False, header=False
)