import pandas as pd
import os

import torch
from torch.utils.data import DataLoader
from torchtune import config as torchtune_config


def input_process(input_arg: str, config: dict) -> pd.DataFrame:
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
        config['cdr3_ex'] = True
        config['epitope_ex'] = True

    else:
        inp_data = pd.read_csv(input_arg)
        config['cdr3_ex'] = False
        config['epitope_ex'] = False

        # Check existanse of columns
        if 'cdr3' in inp_data.columns:
            config['cdr3_ex'] = True
        if 'antigen_epitope' in inp_data.columns:
            config['epitope_ex'] = True

        if config['cdr3_ex'] is False and config['epitope_ex'] is False:
            raise ValueError('Input data should contain "cdr3" or "antigen_epitope" columns.')

    return inp_data


def model_train(model,
                input_dataloader: DataLoader,
                device,
                criterion,
                config: dict,
                test_dataloader: DataLoader = None):
    """
    """

    optimizer = torchtune_config.instantiate(config['OPTIMIZER'], model.parameters())

    if test_dataloader is not None:
        train_test = True
    else:
        train_test = False

    train_losses = []
    test_losses = []
    num_epochs = config['EPOCH_NUM']

    print(f'Training model for {model.seq_type} ...')
    for epoch in range(num_epochs):

        model.train()
        epoch_train_loss = 0.0

        for (pep_batch) in input_dataloader:
            pep_batch_x = pep_batch[0].to(device)

            pep_batch_recon = model(pep_batch_x)

            loss = criterion(pep_batch_recon, pep_batch_x)
            epoch_train_loss += loss.item() * pep_batch_x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = epoch_train_loss / len(input_dataloader.dataset)
        train_losses.append(train_loss)

        if train_test is True:
            model.eval()
            epoch_test_loss = 0.0
            with torch.no_grad():
                for test_pep_batch in test_dataloader:
                    test_pep_batch = test_pep_batch[0].to(device)

                    test_pep_batch_recon = model(test_pep_batch)

                    loss = criterion(test_pep_batch_recon, test_pep_batch)
                    epoch_test_loss += loss.item() * test_pep_batch.size(0)

            test_loss = epoch_test_loss / len(test_dataloader.dataset)
            test_losses.append(test_loss)

        if epoch % 20 == 0:
            if train_test is True:
                print(
                    f"[{epoch}/{num_epochs}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
                )
            else:
                print(
                    f"[{epoch}/{num_epochs}] Train Loss: {train_loss:.4f}"
                )

    print(f'Train for {model.seq_type} finished')


def saving_results(model, output_dir: str, embed_type: str, seqs_type: str) -> None:
    output_path_suffix = f'weights_{embed_type}_{seqs_type}.pth'
    output_path = output_dir.joinpath(output_path_suffix)
    print(output_path)
    torch.save(model.state_dict(), output_path)
