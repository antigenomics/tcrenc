import pandas as pd
import os
import torch
from omegaconf import OmegaConf


def read_config(config_path: str, script_type=None) -> dict:
    """
    Reads and processes configuration file using OmegaConf.

    Args:
        config_path: Path to the configuration file
        script_type: Optional script type to load specific configuration section.
                    If None, only general config is loaded.

    Returns:
        Dictionary containing merged configuration parameters.

    Note:
        - If 'general' is in config_path, loads only that file
        - Otherwise loads general config and merges with script-specific config
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
    """
    Filters input data based on configuration parameters.

    Args:
        inp_data: Input DataFrame containing sequences
        conf: Configuration dictionary with filtering parameters

    Returns:
        Filtered DataFrame with sequences meeting requirements

    Raises:
        ValueError: If no rows satisfy the filtering conditions
    """
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
        raise ValueError('There are no rows in input that satisfy the conditions in config.')

    return filtered_data


def set_device(use_gpu: bool):
    """
    Determines and returns the appropriate computation device.

    Args:
        use_gpu: Boolean flag indicating whether to attempt GPU usage

    Returns:
        torch.device: The selected computation device
        - CUDA GPU if available and use_gpu=True
        - MPS (Apple Metal) if available and use_gpu=True
        - CPU otherwise
    """
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    elif use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def input_process(input_arg: str, gen_config: dict) -> pd.DataFrame:
    """
    Processes input argument and loads data accordingly.

    Args:
        input_arg: Either 'VDJdb' or path to input CSV file
        gen_config: General configuration dictionary that will be updated

    Returns:
        DataFrame containing loaded sequence data

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input lacks required columns

    Note:
        - For VDJdb: loads TRB gene and HomoSapiens species only
        - Updates gen_config with presence flags for sequence types
        - Expected columns: 'cdr3' and/or 'antigen_epitope'
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
