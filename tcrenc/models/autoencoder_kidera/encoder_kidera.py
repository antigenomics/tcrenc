import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf.errors import ConfigKeyError

from tcrenc.models.encoder import Encoder
import tcrenc.utils.constants as constants
from tcrenc.utils.run import model_process
from tcrenc.utils.train import part_model_train, saving_weights


LEN_AA_LIST = len(constants.AA_LIST)
KIDERA_DICT = constants.AA_LIST_KIDERA_FACTORS_scaled


class Encoder_kidera(Encoder):
    """
    Encoder network for protein sequences using Kidera factors.

    Converts amino acid sequences into latent space representations using
    Kidera factors (10-dimensional amino acid features) and convolutional neural networks.
    """
    def __init__(self,
                 config: dict,
                 seq_type: str,
                 device: torch.device):
        """
        Initializes the Kidera encoder.

        Args:
            config: Configuration dictionary containing model parameters
            seq_type: Type of sequence ('cdr3' or 'antigen_epitope')
            device: Torch device for computation (CPU/GPU)

        Raises:
            ValueError: If unknown sequence type is provided
            ConfigKeyError: If weight path is not found in config
        """
        super(Encoder_kidera, self).__init__()

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

        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder network.

        Args:
            x: Input tensor

        Returns:
            Encoded tensor of shape (batch_size, latent_dims)
        """
        return self.linear_encode(self.encoder(x))

    def weight_load(self) -> None:
        """
        Loads pretrained weights for the model.
        """
        self.load_state_dict(torch.load(self.weight_path,
                                        map_location=self.device,
                                        weights_only=True))

    def make_embeddings_from_seq(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates latent embeddings from input sequences.

        Args:
            input_data: DataFrame containing sequences in seq_type column

        Returns:
            DataFrame containing embeddings concatenated with input sequences
        """
        input_dataloader = self.input_data_process(inp_data=input_data[self.seq_type])

        model_output = model_process(self,
                                     inp_dataloader=input_dataloader,
                                     device=self.device)

        embeddings = self.embeddings_data_process(model_output,
                                                  input_data[self.seq_type])
        return embeddings

    def input_data_process(self, inp_data: pd.Series) -> DataLoader:
        """
        Prepares input sequences for model processing.

        Processes sequences by:
        1. Inserting gaps to standardize lengths
        2. Converting to Kidera factor representation
        3. Creating DataLoader for batch processing

        Args:
            inp_data: Pandas Series containing protein sequences

        Returns:
            DataLoader with processed sequences in Kidera factor representation
        """
        data = inp_data.copy()

        data = data.apply(self._gap_insertion)
        data_tensor = torch.tensor(
            np.stack(data
                     .map(lambda seq: self._sequence_to_factor(seq))
                     .values,
                     axis=0,
                     ),
            dtype=torch.float32,
        ).unsqueeze(1)

        inp_dataset = TensorDataset(data_tensor)

        inp_dataloader = DataLoader(inp_dataset,
                                    batch_size=self.config['BATCH_SIZE'],
                                    shuffle=False)
        return inp_dataloader

    def embeddings_data_process(self,
                                encoder_output: torch.Tensor,
                                input_seqs: pd.Series) -> pd.DataFrame:
        """
        Post-processes encoder output into final embeddings.

        Args:
            encoder_output: Tensor of shape (n_samples, latent_dims)
            input_seqs: Original input sequences

        Returns:
            DataFrame with embeddings concatenated with original sequences
        """
        embd = pd.concat([pd.DataFrame(encoder_output),
                          input_seqs.to_frame()],
                         axis=1)
        return embd

    def model_train(self,
                    train_data: pd.DataFrame,
                    criterion,
                    test_data=None):
        """
        Trains the encoder model.

        Args:
            train_data: DataFrame containing training sequences and embeddings
            criterion: Loss function for training
            test_data: Optional DataFrame for validation data

        Raises:
            ValueError: If input data has incorrect shape
        """
        self._embds_shape_check(train_data.drop(columns=self.seq_type))

        if test_data is None:
            seq_train_dataloader = self.input_data_process(inp_data=train_data[self.seq_type])
            seq_test_dataloader = None

            embds_train_dataloader = self._make_embds_dataloader(
                input_embeddings=train_data.drop(columns=self.seq_type)
                )
            embds_test_dataloader = None
        else:
            seq_train_dataloader = self.input_data_process(inp_data=train_data[self.seq_type])
            seq_test_dataloader = self.input_data_process(inp_data=test_data[self.seq_type])

            embds_train_dataloader = self._make_embds_dataloader(
                input_embeddings=train_data.drop(columns=self.seq_type)
                )
            embds_test_dataloader = self._make_embds_dataloader(
                input_embeddings=test_data.drop(columns=self.seq_type)
                )

        part_model_train(self,
                         model_type='encoder',
                         seq_train_dataloader=seq_train_dataloader,
                         embds_train_dataloader=embds_train_dataloader,
                         device=self.device,
                         criterion=criterion,
                         config=self.config,
                         seq_test_dataloader=seq_test_dataloader,
                         embds_test_dataloader=embds_test_dataloader)

    def save_model(self, output_dir: Path) -> None:
        """
        Saves model weights to specified directory.

        Args:
            output_dir: Directory to save model weights
        """
        saving_weights(self, output_dir, self.embd_type, self.seq_type)

    def _gap_insertion(self, inp_seq: str) -> str:

        """
        Inserts gaps to standardize sequence lengths.

        Args:
            inp_seq: Amino acid sequence string

        Returns:
            Sequence with inserted gaps to match max length

        Note:
            Different gap insertion strategies are used for:
            - Epitopes: Center-padded with gaps
            - CDR3 sequences: Specific gap patterns based on length
        """

        if self.seq_type == 'antigen_epitope':
            start_end = (self._max_len - len(inp_seq)) // 2
            if len(inp_seq) % 2 == 0:
                return start_end * "-" + inp_seq + start_end * "-"
            else:
                return (
                    start_end * "-"
                    + inp_seq[: len(inp_seq) // 2]
                    + "-"
                    + inp_seq[len(inp_seq) // 2:]
                    + start_end * "-"
                       )

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

    def _sequence_to_factor(self, sequence: str) -> np.ndarray:
        """
        Converts amino acid sequence to Kidera factors.

        Args:
            sequence: Amino acid sequence string

        Returns:
            2D numpy array of shape (10, sequence_length) containing Kidera factors
        """
        return np.array([KIDERA_DICT[aa] for aa in sequence],
                        dtype=np.float32).T

    def _embds_shape_check(self, data: pd.DataFrame) -> None:
        """
        Validates input embeddings shape.

        Args:
            data: Input embeddings DataFrame

        Raises:
            ValueError: If embeddings have incorrect shape (should match latent_dims)
        """
        if (data.shape[1]) != self.latent_dims:
            raise ValueError('Wrong embeddings shape')

    def _make_embds_dataloader(self, input_embeddings: pd.DataFrame) -> DataLoader:
        """
        Creates DataLoader from input embeddings.

        Args:
            input_embeddings: DataFrame containing embeddings

        Returns:
            DataLoader with embeddings
        """
        embds_np = input_embeddings.to_numpy(dtype=np.float32)
        embds_dataset = TensorDataset(torch.tensor(embds_np))
        embds_dataloader = DataLoader(embds_dataset,
                                      batch_size=self.config['BATCH_SIZE'],
                                      shuffle=False)
        return embds_dataloader
