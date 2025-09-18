import pandas as pd

import torch
from torch.utils.data import DataLoader


def model_process(model, inp_dataloader: DataLoader, device):
    """
    Function to create embeddings by using make_embeddings_from_seq function of model.
    It also prints reconstruction error based on criterion specified in model config file.
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
    Function to save embeddings to csv file.
    One row consist input seq and embedding for it.
    """
    if args.decoder:
        process_result = 'reconstructed'
    else:
        process_result = 'embeddings'

    output_path = f'{output_dir}/{process_result}_{seqs_type}_{args.embed_type}.csv'

    df.to_csv(output_path, index=False)

    print(f'{output_path} file saved!')
