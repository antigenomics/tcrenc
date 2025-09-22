import torch
from torch.utils.data import DataLoader
from torchtune import config as torchtune_config


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

        if train_test:
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
            if train_test:
                print(
                    f"[{epoch}/{num_epochs}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
                )
            else:
                print(
                    f"[{epoch}/{num_epochs}] Train Loss: {train_loss:.4f}"
                )

    print(f'Train for {model.seq_type} finished')


def part_model_train(model,
                     model_type: str,
                     seq_train_dataloader: DataLoader,
                     embds_train_dataloader: DataLoader,
                     device,
                     criterion,
                     config: dict,
                     seq_test_dataloader: DataLoader = None,
                     embds_test_dataloader: DataLoader = None):
    """
    """

    optimizer = torchtune_config.instantiate(config['OPTIMIZER'], model.parameters())

    if seq_test_dataloader is not None:
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

        for (seq_batch, embds_batch) in zip(seq_train_dataloader, embds_train_dataloader):
            seq_batch_x = seq_batch[0].to(device)
            embds_batch_x = embds_batch[0].to(device)

            if model_type == 'encoder':
                embds_batch_reconstructed = model(seq_batch_x)
                loss = criterion(embds_batch_reconstructed, embds_batch_x)
            elif model_type == 'decoder':
                seq_batch_reconstructed = model(embds_batch_x)
                loss = criterion(seq_batch_reconstructed, seq_batch_x)

            epoch_train_loss += loss.item() * seq_batch_x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = epoch_train_loss / len(seq_train_dataloader.dataset)
        train_losses.append(train_loss)

        if train_test:
            model.eval()
            epoch_test_loss = 0.0

            with torch.no_grad():
                for (seq_batch, embds_batch) in zip(seq_test_dataloader, embds_test_dataloader):

                    seq_batch_x = seq_batch[0].to(device)
                    embds_batch_x = embds_batch[0].to(device)

                    if model_type == 'encoder':
                        embds_batch_reconstructed = model(seq_batch_x)
                        loss = criterion(embds_batch_reconstructed, embds_batch_x)
                    elif model_type == 'decoder':
                        seq_batch_reconstructed = model(embds_batch_x)
                        loss = criterion(seq_batch_reconstructed, seq_batch_x)

                    epoch_test_loss += loss.item() * seq_batch_x.size(0)

            test_loss = epoch_test_loss / len(seq_test_dataloader.dataset)
            test_losses.append(test_loss)

        if epoch % 20 == 0:
            if train_test:
                print(
                    f"[{epoch}/{num_epochs}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
                )
            else:
                print(
                    f"[{epoch}/{num_epochs}] Train Loss: {train_loss:.4f}"
                )

    print(f'Train for {model.seq_type} finished')


def saving_results(model, output_dir, embed_type: str, seqs_type: str) -> None:
    output_path_suffix = f'weights_{embed_type}_{seqs_type}.pth'
    output_path = output_dir.joinpath(output_path_suffix)
    print(output_path)
    torch.save(model.state_dict(), output_path)
