# Standard library imports
import argparse
import sys
import warnings
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Path setup
current_file_path = Path(__file__).absolute()
project_root = current_file_path.parent.parent.parent
sys.path.append(str(project_root))

# Local imports
from actual_encoders.autoencoder import ConvAutoEncoder
from modules.modules_kidera.gpu import GPU
from modules.modules_kidera.kidera import kidera_final_dict
from Not_actual_encoders.autoencoder_residual import ConvAutoEncoderRes


# Constants
USE_GPU = True
BATCH_SIZE = 64

# Suppress warnings
warnings.filterwarnings("ignore")


# Initialize device
device = GPU(USE_GPU)


def sequence_to_factor(sequence, kidera_dict):
    """Convert amino acid sequence to Kidera factors."""
    return np.array([kidera_dict[aa] for aa in sequence], dtype=np.float32).T


def insert_x(x: str):
    """
    Insert fictitious amino acid (X) to make all sequences with length 19.
    
    Args:
        x: Amino acid sequence
    """
    length_x = 19 - len(x)
    if len(x) == 4:
        return x[:2] + "X" * 15 + x[2:]
    elif len(x) == 5:
        return x[:2] + "X" * 7 + x[2] + "X" * 7 + x[3:]
    elif len(x) == 6:
        return x[:3] + "X" * 13 + x[3:]
    else:
        pref, suff = x[:3], x[-3:]
        mid = x[3:-3]
        return (
            pref
            + "X" * (length_x // 2 + length_x % 2)
            + mid
            + "X" * (length_x // 2)
            + suff
        )


def func_antigen(antigen: str):
    """
    Insert fictitious amino acid (X) to make all sequences with length 20.
    
    Args:
        antigen: Amino acid sequence
    """
    n = 20
    start_end = (n - len(antigen)) // 2
    if len(antigen) % 2 == 0:
        return start_end * "X" + antigen + start_end * "X"
    else:
        return (
            start_end * "X"
            + antigen[: len(antigen) // 2]
            + "X"
            + antigen[len(antigen) // 2:]
            + start_end * "X"
        )


def epitope_to_kidera(epitopes: pd.Series, kidera_dict: pd.DataFrame):
    """Convert epitope sequences to Kidera factors tensor."""
    epitopes = epitopes.apply(func_antigen)
    factors_array = np.stack(
        epitopes.map(lambda seq: sequence_to_factor(seq, kidera_dict)).values,
        axis=0
    )
    return torch.tensor(factors_array, dtype=torch.float32).unsqueeze(1)


def get_encoded_data(data_enc, model, batch_size, device="cuda"):
    """
    Pass encodings through an autoencoder and return both encoded and decoded outputs.
    
    Args:
        data_enc: Input tensor of encoded data
        model: Trained autoencoder model
        batch_size: Batch size for processing
        device: Device for computation ('cuda' or 'cpu')
        
    Returns:
        Tuple of (encoded representations, reconstructed data)
    """
    model.eval()
    model.to(device)
    test_loader = DataLoader(TensorDataset(data_enc), batch_size=batch_size)
    encoded_data, decoded_data = [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            latent = model.linear_encode(model.encoder(x))
            decoded = model(x)
            encoded_data.append(latent.cpu())
            decoded_data.append(decoded.cpu())
            
    return torch.cat(encoded_data), torch.cat(decoded_data)


def process(input_file: str, output_dir: str, residual_block: bool) -> None:
    """Main processing function."""
    # Load and preprocess data
    data = pd.read_csv(input_file)
    data["cdr3"] = data["cdr3"].apply(insert_x)
    data_cdr3 = data[["cdr3"]]
    epitopes = data[["antigen_epitope"]]

    # Prepare tensors
    data_train_test_cdr3 = torch.tensor(
        np.stack(
            data_cdr3["cdr3"]
            .map(lambda seq: sequence_to_factor(seq, kidera_final_dict))
            .values,
            axis=0,
        ),
        dtype=torch.float32,
    ).unsqueeze(1)

    epitope_tensor = epitope_to_kidera(
        epitopes["antigen_epitope"], 
        kidera_final_dict
    )

    if residual_block:
        # Process with residual blocks
        model_cdr3 = ConvAutoEncoderRes(linear=19, latent_dim=64).to(device)
        model_cdr3.load_state_dict(
            torch.load("/projects/tcr_nlp/conv_autoencoder/conv_res_block/cdr3_res.pth")
        )
        
        model_epitope = ConvAutoEncoderRes(linear=20, latent_dim=64).to(device)
        model_epitope.load_state_dict(
            torch.load("/projects/tcr_nlp/conv_autoencoder/conv_res_block/epitope_res.pth")
        )
        
        output_suffix = "_residual"
    else:
        # Process without residual blocks
        model_cdr3 = ConvAutoEncoder(linear=19, latent_dim=64).to(device)
        model_cdr3.load_state_dict(
            torch.load("/projects/tcr_nlp/conv_autoencoder/conv/cdr3.pth")
        )
        
        model_epitope = ConvAutoEncoder(linear=20, latent_dim=64).to(device)
        model_epitope.load_state_dict(
            torch.load("/projects/tcr_nlp/conv_autoencoder/conv/epitope.pth")
        )
        
        output_suffix = ""

    # Process and save CDR3 data
    encoded_cdr3, _ = get_encoded_data(
        data_train_test_cdr3, 
        model_cdr3, 
        BATCH_SIZE
    )
    pd.DataFrame(encoded_cdr3).to_csv(
        f"{output_dir}/encoded_cdr3_kidera{output_suffix}.csv",
        index=False,
        header=False
    )

    # Process and save epitope data
    encoded_epitope, _ = get_encoded_data(
        epitope_tensor,
        model_epitope,
        BATCH_SIZE
    )
    pd.DataFrame(encoded_epitope).to_csv(
        f"{output_dir}/encoded_epitope_kidera{output_suffix}.csv",
        index=False,
        header=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--residual_block", type=str, required=True)

    args = parser.parse_args()
    process(
        args.input,
        args.output,
        args.residual_block.lower() == "true"
    )
