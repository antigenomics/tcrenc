import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor, tensor
from torch.utils.data import DataLoader, TensorDataset

from tcrenc.models.autoencoder import Autoencoder
import tcrenc.utils.constants as constants


LEN_AA_LIST = len(constants.AA_LIST)
KIDERA_DICT = constants.AA_LIST_KIDERA_FACTORS_scaled


# NOW ONLY FOR CDR3 (for epitope different lenght)


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
                 seq_type: str):
        """
        Initializes the convolutional autoencoder.

        Args:
            ... TODO
        """
        super(Autoencoder_kidera, self).__init__()

        self.config = config
        self.seq_type = seq_type

        if self.seq_type == 'cdr3':
            self._max_len = self.config['MAX_CDR3_LEN']
            self.input_dims = self._max_len * LEN_AA_LIST
        elif self.seq_type == 'antigen_epitope':
            self._max_len = self.config['MAX_EPITOPE_LEN']
            self.input_dims = self._max_len * LEN_AA_LIST
        else:
            raise ValueError('Unknown seq type for this model.')

        self.latent_dims = self.config['LATENT_DIMS']
        self.linear_part = self.config['LINEAR_PART']

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear_encode = nn.Sequential(
            nn.Linear(128*10*self.linear_part, 1024), nn.ReLU(), nn.Linear(1024, self.latent_dims)
        )

        self.linear_decode = nn.Sequential(
            nn.Linear(self.latent_dims, 1024), nn.ReLU(), nn.Linear(1024, 128*10*self.linear_part)
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 10, self.linear_part)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inp_seq: Tensor) -> Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            inp_seq (torch.Tensor): Input tensor of shape (batch_size, 1, 10, linear),
                              where 10 is the number of features per amino acid,
                              and linear is the number of amino acids.

        Returns:
            torch.Tensor: Reconstructed input tensor of the same shape.
        """
        encoded = self.linear_encode(self.encoder(inp_seq))
        decoded = self.decoder(self.linear_decode(encoded))
        return decoded

    def make_embeddings_from_seq(self, inp_seq: Tensor) -> Tensor:
        encoded = self.linear_encode(self.encoder(inp_seq))
        return encoded

    def make_seq_from_embeddings(self, encoded: Tensor) -> Tensor:
        decoded = self.decoder(self.linear_decode(encoded))
        return decoded

    def _gap_insertion(self, inp_seq: str) -> str:

        """
        Insert gaps (-) to make all sequences with max length
        Args:
            inp_seq: Amino acid sequence
        """

        if self.seq_type == 'cdr3':
            # TODO
            pass

        length_x = self._max_len - len(inp_seq)
        if len(inp_seq) == 4:
            return inp_seq[:2] + "-" * 15 + inp_seq[2:]
        elif len(inp_seq) == 5:
            return inp_seq[:2] + "-" * 7 + inp_seq[2] + "-" * 7 + inp_seq[3:]
        elif len(inp_seq) == 6:
            return inp_seq[:3] + "-" * 13 + inp_seq[3:]
        else:
            pref, suff = inp_seq[:3], inp_seq[-3:]
            mid = inp_seq[3:-3]
            return (
                pref
                + "-" * (length_x // 2 + length_x % 2)
                + mid
                + "-" * (length_x // 2)
                + suff
            )

    def _sequence_to_factor(self, sequence: str) -> np.array:
        """Convert amino acid sequence to Kidera factors."""
        return np.array([KIDERA_DICT[aa] for aa in sequence], dtype=np.float32).T

    def input_data_process(self, inp_data: pd.Series) -> DataLoader:
        """
        Main function to prepare torch DataLoader for input pandas Series, consist of 'cdr3' or 'antigen_epitope' sequences.
        It add gaps and ... TODO description
        """
        data = inp_data.copy()
        data = data.apply(self._gap_insertion)
        data_tensor = tensor(
            np.stack(data
                     .map(lambda seq: self._sequence_to_factor(seq))
                     .values,
                     axis=0,
                     ),
            dtype=torch.float32,
        ).unsqueeze(1)

        inp_dataloader = DataLoader(TensorDataset(data_tensor),
                                    batch_size=self.config['BATCH_SIZE'])

        return inp_dataloader

    def reconstructed_data_process(self, reconstructed_data: list) -> list:
        # TODO make this function
        pass

    def embeddings_data_process(self, encoder_output: Tensor, input_seqs: pd.Series) -> pd.DataFrame:

        embd = pd.concat([pd.DataFrame(encoder_output),
                          input_seqs.to_frame()],
                         axis=1)

        return embd
