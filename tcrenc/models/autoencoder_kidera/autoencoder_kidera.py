import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf.errors import ConfigKeyError
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from tcrenc.models.autoencoder import Autoencoder
from tcrenc.models.autoencoder_kidera.decoder_kidera import Decoder_kidera
from tcrenc.models.autoencoder_kidera.encoder_kidera import Encoder_kidera
import tcrenc.utils.constants as constants
from tcrenc.utils.validate import model_validate
from tcrenc.utils.train import model_train, saving_weights


LEN_AA_LIST = len(constants.AA_LIST)
KIDERA_DICT = constants.AA_LIST_KIDERA_FACTORS_scaled


class Autoencoder_kidera(Autoencoder):
    """
    Convolutional Autoencoder for encoded CDR3 amino acid sequences.

    This model is designed to learn compact latent representations of CDR3 sequences
    by using convolutional and fully connected layers.
    It operates on numerical feature representations of amino acids (kidera factors) shaped as 2D tensors.

    Args:
        linear (int): The length of the CDR3 sequence (or adjusted width after encoding),
                      which determines the spatial dimension of the convolutional input.
        latent_dim (int, optional): Size of the latent space vector. Default is 64.

    Architecture:
        - Encoder:
            3 convolutional layers with batch normalization and ReLU, followed by flattening.
        - Latent Projection:
            A bottleneck of fully connected layers maps the high-dimensional features into
            a lower-dimensional latent space.
        - Decoder:
            The latent vector is reconstructed through linear layers, reshaped, and passed
            through transposed convolutions to reconstruct the input.

    Note:
        - Input tensor shape must be (batch_size, 1, 10, linear), where `10` is the number of
          amino acid features per residue, and `linear` is the number of
          amino acids in the CDR3 sequence (possibly padded to a fixed length).
    """

    def __init__(self,
                 config: dict,
                 seq_type: str,
                 device: torch.device):
        """
        Initializes the convolutional autoencoder.

        Args:
            ... TODO
        """
        super(Autoencoder_kidera, self).__init__()

        self.config = config
        self.seq_type = seq_type
        self.embd_type = 'kidera'

        if self.seq_type == 'cdr3':
            self._max_len = self.config['MAX_CDR3_LEN']
            self.input_dims = self._max_len * LEN_AA_LIST
            self.linear_part = self.config['LINEAR_PART_CDR3']
            try:
                self.weight_path = self.config['WEIGHTS_CDR3']
            except ConfigKeyError:
                self.weight_path = None

        elif self.seq_type == 'antigen_epitope':
            self._max_len = self.config['MAX_EPITOPE_LEN']
            self.input_dims = self._max_len * LEN_AA_LIST
            self.linear_part = self.config['LINEAR_PART_EPITOPE']
            try:
                self.weight_path = self.config['WEIGHTS_EPIOPE']
            except ConfigKeyError:
                self.weight_path = None

        else:
            raise ValueError('Unknown seq type for this model.')

        self.latent_dims = self.config['LATENT_DIMS']

        self.encoder = Encoder_kidera(config=self.config,
                                      seq_type=self.seq_type,
                                      device=device)

        self.decoder = Decoder_kidera(config=self.config,
                                      seq_type=self.seq_type,
                                      device=device)
        self.device = device
        self.umap = None
        self.rfc = None

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            inp_seq (torch.torch.Tensor): Input tensor of shape (batch_size, 1, 10, linear),
                              where 10 is the number of features per amino acid,
                              and linear is the number of amino acids.

        Returns:
            torch.torch.Tensor: Reconstructed input tensor of the same shape.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def weight_load(self) -> None:
        """
        """
        self.load_state_dict(torch.load(self.weight_path,
                                        map_location=self.device,
                                        weights_only=True))
        # Make path to umap
        directory = os.path.dirname(self.weight_path)

        umap_filename = f'umap_{self.embd_type}_{self.seq_type}.pkl'
        umap_path = os.path.join(directory, umap_filename)
        with open(umap_path, "rb") as f:
            self.umap = pickle.load(f)
            self.decoder.umap = self.umap

        rfc_filename = f'rfc_{self.embd_type}_{self.seq_type}.pkl'
        rfc_path = os.path.join(directory, rfc_filename)
        with open(rfc_path, "rb") as f:
            self.rfc = pickle.load(f)
            self.decoder.rfc = self.rfc

    def input_data_process(self, inp_seqs: pd.Series) -> DataLoader:
        """
        Main function to prepare torch DataLoader for input pandas Series, consist of 'cdr3' or 'antigen_epitope' sequences.
        It add gaps and ... TODO description
        """
        inp_dataloader = self.encoder.input_data_process(inp_data=inp_seqs)
        return inp_dataloader

    def make_embeddings_from_seq(self, input_data: pd.DataFrame) -> pd.DataFrame:
        embeddings = self.encoder.make_embeddings_from_seq(input_data=input_data)
        return embeddings

    def make_seq_from_embeddings(self, input_embds: pd.DataFrame) -> pd.DataFrame:
        decoded_seqs = self.decoder.make_seq_from_embeddings(input_embds=input_embds)
        return decoded_seqs

    def reconstructed_data_process(self, input_tensor: torch.Tensor) -> pd.DataFrame:
        reconstructed_data = self.decoder.reconstructed_data_process(reconstructed_data=input_tensor)
        return reconstructed_data

    def model_train(self,
                    train_data: DataLoader,
                    criterion,
                    input_train_seqs: pd.Series,
                    test_data=None) -> None:

        input_dataloader = self.input_data_process(inp_seqs=train_data[self.seq_type])

        if test_data is not None:
            test_dataloader = self.input_data_process(inp_seqs=test_data[self.seq_type])
        else:
            test_dataloader = test_data

        model_train(self,
                    input_dataloader=input_dataloader,
                    device=self.device,
                    criterion=criterion,
                    config=self.config,
                    test_dataloader=test_dataloader)

        _, output_seqs_coded, _ = model_validate(self,
                                                 input_dataloader=input_dataloader,
                                                 device=self.device,
                                                 criterion=criterion)

        print(f"Start UMAP training for {self.seq_type}\n\n")

        labels = {x: i for i, x in enumerate(KIDERA_DICT.keys())}

        input_train_seqs = input_train_seqs.apply(self.encoder._gap_insertion)
        inp_seqs = input_train_seqs.str.cat(sep='')

        model_out_reshaped_labeled = []
        for num, seq_tensor in enumerate(output_seqs_coded):
            seq = seq_tensor.squeeze(0).T
            model_out_reshaped_labeled.extend(
                  np.column_stack([seq,
                                   np.array(
                                       [labels[inp_seqs[num*self._max_len + x]] for x in range(self._max_len)]
                                       )]))

        df = pd.DataFrame(model_out_reshaped_labeled)
        df[10] = df[10].astype(int)
        target = df[10]
        df_sampled = df.groupby(10, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), 20000), random_state=42)
        )
        df_sampled.drop(columns=10, inplace=True)
        df.drop(columns=10, inplace=True)

        self.umap = UMAP(n_neighbors=10, min_dist=0.3, n_jobs=-1,
                         verbose=False).fit(df_sampled)

        umap_embedding = self.umap.transform(df)
        print(f"UMAP training for {self.seq_type} finished!\n\n")

        print(f"Start RandomForestClassifier training for {self.seq_type}\n\n")
        X_train, X_test, y_train, y_test = train_test_split(
            umap_embedding, target, test_size=0.3, random_state=42
        )

        self.rfc = RandomForestClassifier(n_estimators=20)
        self.rfc.fit(X_train, y_train)
        print(f"RandomForestClassifier training for {self.seq_type} finished!\n\n")

        print("Classification report for full model:\n")
        print(classification_report(y_test, self.rfc.predict(X_test)))

        self.decoder.umap = self.umap
        self.decoder.rfc = self.rfc

    def save_model(self, output_dir: Path) -> None:
        """
        """
        saving_weights(self, output_dir, self.embd_type, self.seq_type)

        umap_suffix = f'umap_{self.embd_type}_{self.seq_type}.pkl'
        umap_path = output_dir.joinpath(umap_suffix)
        with open(umap_path, "wb") as f:
            pickle.dump(self.umap, f)

        rfc_suffix = f'rfc_{self.embd_type}_{self.seq_type}.pkl'
        rfc_path = output_dir.joinpath(rfc_suffix)
        with open(rfc_path, "wb") as f:
            pickle.dump(self.rfc, f)

    def validation_on_seqs(self, input_data: pd.DataFrame, loss_function):
        """
        """
        input_dataloader = self.input_data_process(inp_seqs=input_data[self.seq_type])

        _, output_seqs_coded, loss_value = model_validate(self,
                                                          input_dataloader=input_dataloader,
                                                          device=self.device,
                                                          criterion=loss_function)

        input_seqs = input_data[self.seq_type].to_frame()
        output_seqs = self.reconstructed_data_process(output_seqs_coded)

        return input_seqs, output_seqs, loss_value
