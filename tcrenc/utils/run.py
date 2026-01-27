import pandas as pd

import torch
from torch.utils.data import DataLoader


def model_process(model, inp_dataloader: DataLoader, device):
    """
    Processes input data through the model to generate embeddings or reconstructed sequences.

    Args:
        model: The NN PyTorch model to process the data
        inp_dataloader: DataLoader containing input
        device: Computation PyTorch device to use

    Returns:
        torch.Tensor: Concatenated output tensors from the model

    Note:
        - Sets model to evaluation mode
        - Processes data without gradient computation
        - Moves output to CPU memory
        - You can use it for NN models
    """
    model.eval()

    output = []

    for pre_pep in inp_dataloader:
        with torch.no_grad():
            pep = pre_pep[0].to(device)
            pep_encod = model(pep)
        output.append(pep_encod.cpu())

    return torch.cat(output)


def saving_results(df: pd.DataFrame, output_dir: str, args, seqs_type: str) -> None:
    """
    Saves processed results (embeddings or reconstructed sequences) to CSV file.

    Args:
        df: DataFrame containing the results to save
        output_dir: Directory path to save the output file
        args: Command line arguments object containing:
        seqs_type: Type of sequences processed ('cdr3' or 'antigen_epitope')

    Returns:
        None

    Output:
        Creates a CSV file with naming convention:
        - embeddings_[seqs_type]_[embed_type].csv when using encoder
        - reconstructed_[seqs_type]_[embed_type].csv when using decoder

    Note:
        Prints confirmation message when file is saved
    """
    if args.decoder:
        process_result = 'reconstructed'
    else:
        process_result = 'embeddings'

    output_path = f'{output_dir}/{process_result}_{seqs_type}_{args.embed_type}.csv'

    df.to_csv(output_path, index=False)

    print(f'{output_path} file saved!')
