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


def saving_results(embd: pd.DataFrame, output_dir: str, embed_type: str, seqs_type: str) -> None:
    """
    Function to save embeddings to csv file.
    One row consist input seq and embedding for it.
    """
    embd.to_csv(f'{output_dir}/embeddings_{seqs_type}_{embed_type}.csv',
                index=False)

    print(f'{output_dir}/embeddings_{seqs_type}_{embed_type}.csv file saved!')
