import pandas as pd

import torch
from torch.utils.data import DataLoader


def model_process(model, input_dataloader: DataLoader, device,  criterion):
    """
    Function to create embeddings by using make_embeddings_from_seq function of model.
    It also prints reconstruction error based on criterion specified in model config file.
    """
    model.eval()

    output = []
    loss_avg, num_batches = 0, 0

    for pre_pep in input_dataloader:
        with torch.no_grad():
            pep = pre_pep[0].to(device)
            pep_encod = model.make_embeddings_from_seq(pep)
            pep_recon = model(pep)
            loss = criterion(pep_recon, pep)
            loss_avg += loss.item()
            num_batches += 1
        output.append(pep_encod.cpu())
    loss_avg /= num_batches
    print(f'Average reconstruction error of {model.seq_type} sequences on sample: {loss_avg:.4f}')

    return torch.cat(output)


def saving_results(embd: pd.DataFrame, output_dir: str, embed_type: str, seqs_type: str) -> None:
    """
    Function to save embeddings to csv file.
    One row consist input seq and embedding for it.
    """
    embd.to_csv(f'{output_dir}/embeddings_{seqs_type}_{embed_type}.csv',
                index=False)

    print(f'{output_dir}/embeddings_{seqs_type}_{embed_type}.csv file saved!')
