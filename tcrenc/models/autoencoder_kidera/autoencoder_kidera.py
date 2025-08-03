import pandas as pd

import torch
from torch.utils.data import DataLoader

from tcrenc.models.autoencoder import Autoencoder
from tcrenc.models.autoencoder_kidera.decoder_kidera import Decoder_kidera
from tcrenc.models.autoencoder_kidera.encoder_kidera import Encoder_kidera
import tcrenc.utils.constants as constants


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
            self.linear_part = self.config['LINEAR_PART_CDR3']
        elif self.seq_type == 'antigen_epitope':
            self._max_len = self.config['MAX_EPITOPE_LEN']
            self.input_dims = self._max_len * LEN_AA_LIST
            self.linear_part = self.config['LINEAR_PART_EPITOPE']
        else:
            raise ValueError('Unknown seq type for this model.')

        self.latent_dims = self.config['LATENT_DIMS']

        self.encoder = Encoder_kidera(config=self.config, seq_type=self.seq_type)
        self.decoder = Decoder_kidera(config=self.config, seq_type=self.seq_type)

    def forward(self, inp_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            inp_seq (torch.torch.Tensor): Input tensor of shape (batch_size, 1, 10, linear),
                              where 10 is the number of features per amino acid,
                              and linear is the number of amino acids.

        Returns:
            torch.torch.Tensor: Reconstructed input tensor of the same shape.
        """
        encoded = self.encoder(inp_seq)
        decoded = self.decoder(encoded)
        return decoded

    def make_embeddings_from_seq(self, inp_seq: torch.Tensor) -> torch.Tensor:
        return self.encoder(inp_seq)

    def make_seq_from_embeddings(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.decoder(encoded)

    def model_process(self,
                      input_dataloader: DataLoader,
                      device: torch.device,
                      criterion,
                      process_type: str) -> torch.Tensor:

        if process_type == 'train':
            # from tcrenc.utils.train import model_process
            pass

        elif process_type == 'validate':
            # rom tcrenc.utils.validate import model_process
            pass

        elif process_type == 'run':
            from tcrenc.utils.run import model_process
            model_output = model_process(self,
                                         input_dataloader=input_dataloader,
                                         device=device,
                                         criterion=criterion,
                                         )
            return model_output

        else:
            raise ValueError('Unknown process type')

    def input_data_process(self, inp_data: pd.Series) -> DataLoader:
        """
        Main function to prepare torch DataLoader for input pandas Series, consist of 'cdr3' or 'antigen_epitope' sequences.
        It add gaps and ... TODO description
        """
        inp_dataloader = self.encoder.input_data_process(inp_data=inp_data)

        return inp_dataloader

    def reconstructed_data_process(self, reconstructed_data: list) -> list:
        # TODO make this function
        pass

    def embeddings_data_process(self, encoder_output: torch.Tensor, input_seqs: pd.Series) -> pd.DataFrame:
        """
        """
        embd = self.encoder.embeddings_data_process(encoder_output=encoder_output,
                                                    input_seqs=input_seqs)

        return embd
