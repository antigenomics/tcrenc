import pandas as pd
import os
import torch
from omegaconf import OmegaConf


def read_config(config_path: str, script_type=None) -> dict:
    """
    Reading config with omegaconf.
    """
    if 'general' in config_path:
        with open(config_path, 'r') as f:
            config = OmegaConf.load(f)
    else:
        with open(config_path, 'r') as f:
            config_full = OmegaConf.load(f)
        config = config_full['general']
        if script_type is not None:
            config.update(config_full[script_type])

    return config


def filter_input(inp_data: pd.DataFrame, conf: dict) -> pd.DataFrame:

    # Filter CDR3
    filtered_data = inp_data.copy()
    col_names = filtered_data.columns
    filtered_data.dropna(inplace=True, ignore_index=True)

    if 'cdr3' in col_names:
        filtered_data = filtered_data[filtered_data['cdr3'].str.match(r'^C.*[FW]$')]
        filtered_data = filtered_data[filtered_data['cdr3'].str.len() >= conf['MIN_CDR3_LEN']]
        filtered_data = filtered_data[filtered_data['cdr3'].str.len() <= conf['MAX_CDR3_LEN']]

    # Filter Epitope
    if 'antigen_epitope' in col_names:
        filtered_data = filtered_data[filtered_data['antigen_epitope'].str.len() >= conf['MIN_EPITOPE_LEN']]
        filtered_data = filtered_data[filtered_data['antigen_epitope'].str.len() <= conf['MAX_EPITOPE_LEN']]

    filtered_data.reset_index(drop=True, inplace=True)

    if filtered_data.shape[0] == 0:
        # TODO maybe fix
        raise ValueError('There are no rows in input that satisfy the conditions in config.')

    return filtered_data


def set_device(use_gpu: bool):

    '''
    This function allows you using GPU systems. If you have MAC M1 you will use mps system.
    Arguments:
        - use_gpu (bool) - True if you want to use GPU.
    '''
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    elif use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def input_process(input_arg: str, gen_config: dict) -> pd.DataFrame:
    """
    TODO mb move to basic

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
