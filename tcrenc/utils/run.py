import pandas as pd
import os

import torch
from torch.utils.data import DataLoader


def input_process(input_arg: str, gen_config: dict) -> pd.DataFrame:
    """
    This function processes input.
    1. Checks file existance or 'VDJdb' option.
    2. For 'VDJdb': fetchs current version and takes only 'TRB' gene and 'HomoSapiens' species as input.
    3. Updates config dictionary with information about existance  seq types.
    """
    if input_arg != 'VDJdb' and not os.path.isfile(input_arg):
        raise FileNotFoundError(f"Input file {input_arg} does not exist")

    elif input_arg == 'VDJdb':
        inp_data = pd.read_csv('./dataset/vdjdb-2024-11-27-fixed/vdjdb.slim.txt', sep='\t')
        inp_data = inp_data[(inp_data.gene == 'TRB') & (inp_data.species == 'HomoSapiens')]
        inp_data.columns = inp_data.columns.str.replace('.', '_')
        inp_data = inp_data[['cdr3', 'antigen_epitope']]
        inp_data.reset_index(drop=True, inplace=True)
        gen_config['cdr3_ex'] = True
        gen_config['epitope_ex'] = True

    else:
        inp_data = pd.read_csv(input_arg)
        gen_config['cdr3_ex'] = False
        gen_config['epitope_ex'] = False

        # Check existanse of columns
        if 'cdr3' in inp_data.columns:
            gen_config['cdr3_ex'] = True
        if 'antigen_epitope' in inp_data.columns:
            gen_config['epitope_ex'] = True

        if gen_config['cdr3_ex'] is False and gen_config['epitope_ex'] is False:
            raise ValueError('Input data should contain "cdr3" or "antigen_epitope" columns.')

    return inp_data


def model_process(input_dataloader: DataLoader, device, model, criterion):
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
    print('Average reconstruction error of cdr3 sequences on sample: %f' % (loss_avg))

    return torch.cat(output)


def saving_results(output: torch.Tensor, output_path: str, embed_type: str, config: dict, input_seqs: pd.Series) -> None:
    """
    Function to save embeddings to csv file.
    One row consist input seq and embedding for it.
    """

    if embed_type == 'onehot':
        embeddings = output.reshape(input_seqs.shape[0], 4*config['LATENT_DIMS'])
    else:
        embeddings = output

    embd = pd.concat([pd.DataFrame(embeddings),
                      input_seqs.to_frame()],
                     axis=1)
    embd.to_csv(f'{output_path}/embeddings_{input_seqs.name}_{embed_type}.csv',
                index=False)

    print(f'{output_path}/embeddings_{input_seqs.name}_{embed_type}.csv file saved!')
