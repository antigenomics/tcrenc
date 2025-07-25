import pandas as pd
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
    if conf['cdr3_ex'] is True:
        filtered_data = filtered_data[filtered_data['cdr3'].str.match(r'^C.*[FW]$')]
        filtered_data = filtered_data[filtered_data['cdr3'].str.len() >= conf['MIN_CDR3_LEN']]
        filtered_data = filtered_data[filtered_data['cdr3'].str.len() <= conf['MAX_CDR3_LEN']]

    # Filter Epitope
    if conf['epitope_ex'] is True:
        filtered_data = filtered_data[filtered_data['antigen_epitope'].str.len() >= conf['MIN_EPITOPE_LEN']]
        filtered_data = filtered_data[filtered_data['antigen_epitope'].str.len() <= conf['MAX_EPITOPE_LEN']]

    filtered_data.reset_index(drop=True, inplace=True)

    if filtered_data.shape[0] == 0:
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
