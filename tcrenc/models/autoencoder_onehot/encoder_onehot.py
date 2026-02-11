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


class Encoder_onehot(Encoder):
    """One-hot encoder for protein sequences.

    Inherits from base Encoder class and implements functionality for encoding
    amino acid sequences into latent space representations using one-hot encoding.
    """
    def __init__(self,
                 config: dict,
                 seq_type: str,
                 device: torch.device):
        """
        Initializes the one-hot encoder.

        Args:
            config: Configuration dictionary containing model parameters
            seq_type: Type of sequence ('cdr3' or 'antigen_epitope')
            device: Torch device for computation (CPU/GPU)

        Raises:
            ValueError: If unknown sequence type is provided
            ConfigKeyError: If weight path is not found in config
        """
        super(Encoder_onehot, self).__init__()

        self.config = config
        self.seq_type = seq_type
        self.embd_type = 'onehot'

        if self.seq_type == 'cdr3':
            self._max_len = self.config['MAX_CDR3_LEN']
            self.input_dims = self._max_len * LEN_AA_LIST
            try:
                self.weight_path = self.config['WEIGHTS_CDR3']
            except ConfigKeyError:
                self.weight_path = None

        elif self.seq_type == 'antigen_epitope':
            self._max_len = self.config['MAX_EPITOPE_LEN']
            self.input_dims = self._max_len * LEN_AA_LIST
            try:
                self.weight_path = self.config['WEIGHTS_EPIOPE']
            except ConfigKeyError:
                self.weight_path = None

        else:
            raise ValueError('Unknown seq type for this model.')

        self.latent_dims = self.config['LATENT_DIMS']

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_dims,
                      out_features=self.latent_dims),
        )

        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder network.

        Args:
            x: Input tensor of shape (batch_size, LEN_AA_LIST, seq_length)

        Returns:
            Encoded tensor of shape (batch_size, latent_dims)
        """
        return self.encoder(x)

    def weight_load(self) -> None:
        """
        Loads pretrained weights for the model.
        """
        self.load_state_dict(torch.load(self.weight_path,
                                        map_location=self.device,
                                        weights_only=True))

    def make_embeddings_from_seq(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create embeddings from input biological sequences.

        Args:
            input_data (pd.DataFrame): A DataFrame containing a single column
                with one type of sequences.

        Returns:
            DataFrame containing embeddings concatenated with input sequences
        """
        input_dataloader = self.input_data_process(inp_seqs=input_data[self.seq_type])

        model_output = model_process(self,
                                     inp_dataloader=input_dataloader,
                                     device=self.device)

        embeddings = self.embeddings_data_process(model_output,
                                                  input_data[self.seq_type])
        return embeddings

    def input_data_process(self, inp_seqs: pd.Series) -> DataLoader:
        """
        Prepares input sequences for model.

        Processes sequences by:
        1. Inserting gaps at specific positions (+3, +4, -3, -4) (each sequence - four variants)
        2. Converting to one-hot encoding
        3. Creating DataLoader for batch processing

        Args:
            inp_seqs: Pandas Series containing input sequences

        Returns:
            DataLoader with one-hot encoded sequences
        """
        inp_list = inp_seqs.to_list()

        # Extend input list with seq's with gaps
        inp_list_with_gaps = self._gap_insertion(inp_list)

        inp_list_oh = np.zeros((len(inp_list_with_gaps),
                                LEN_AA_LIST,
                                self._max_len),
                               dtype=np.float32)

        # List of seqs with gaps to one-hot representation
        for idx, seq in enumerate(inp_list_with_gaps):
            inp_list_oh[idx] = self._one_hot_code(seq)

        inp_dataset = TensorDataset(torch.tensor(inp_list_oh))

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
            encoder_output: Tensor of shape (n_samples*4, latent_dims)
            input_seqs: Original input sequences

        Returns:
            DataFrame with embeddings concatenated with original sequences (n_samples, latent_dims*4 + 1)
        """
        embeddings = encoder_output.reshape(input_seqs.shape[0], 4*self.config['LATENT_DIMS'])

        embd = pd.concat([pd.DataFrame(embeddings),
                          input_seqs.to_frame()],
                         axis=1)
        return embd

    def model_train(self,
                    train_data: pd.DataFrame,
                    criterion,
                    test_data=None) -> None:
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
            seq_train_dataloader = self.input_data_process(inp_seqs=train_data[self.seq_type])
            seq_test_dataloader = None

            embds_train_dataloader = self._make_embds_dataloader(
                input_embeddings=train_data.drop(columns=self.seq_type)
                )
            embds_test_dataloader = None
        else:
            seq_train_dataloader = self.input_data_process(inp_seqs=train_data[self.seq_type])
            seq_test_dataloader = self.input_data_process(inp_seqs=test_data[self.seq_type])

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

    def save_model(self, output_path: Path) -> None:
        """
        Saves model weights to specified path.

        Args:
            output_path: Path to save model weights
        """
        saving_weights(self, output_path, self.embd_type, self.seq_type)

    def _gap_insertion(self, inp_list: list) -> list:
        """
        Inserts gaps into sequences of cdr3 and epitope to positions +3, +4, -3, -4.

        Args:
            inp_list: List of input sequences

        Returns:
            List of sequences with inserted gaps
        """
        ext_list = []

        for seq in inp_list:
            gap_count = self._max_len - len(seq)

            ext_list.append(seq[0:3]+'-'*gap_count+seq[3:])
            ext_list.append(seq[0:4]+'-'*gap_count+seq[4:])
            ext_list.append(seq[0:-3]+'-'*gap_count+seq[-3:])
            ext_list.append(seq[0:-4]+'-'*gap_count+seq[-4:])

        return ext_list

    def _one_hot_code(self, peptide: str) -> np.ndarray:
        """
        Encodes peptide sequence into one-hot matrix.

        Args:
            peptide: Amino acid sequence string

        Returns:
            2D numpy array of shape (LEN_AA_LIST, seq_length)
        """
        pep_oh_encoded = np.zeros((LEN_AA_LIST, len(peptide)),
                                  dtype=np.float32)

        for idx, aa in enumerate(peptide):
            aa_idx = constants.AA_LIST.index(aa)
            pep_oh_encoded[aa_idx][idx] = 1

        return pep_oh_encoded

    def _embds_shape_check(self, embds_inp_data: pd.DataFrame) -> None:
        """
        Validates input embeddings shape.

        Args:
            embds_inp_data: Input embeddings DataFrame

        Raises:
            ValueError: If embeddings have incorrect shape (should be 4*latent_dims)
        """
        if (embds_inp_data.shape[1]) != 4 * self.latent_dims:
            raise ValueError('Wrong embeddings shape')

    def _make_embds_dataloader(self, input_embeddings: pd.DataFrame) -> DataLoader:
        """
        Creates DataLoader from input embeddings.

        Args:
            input_embeddings: DataFrame containing embeddings

        Returns:
            DataLoader with reshaped embeddings
        """
        embds_np = input_embeddings.to_numpy(dtype=np.float32)
        embds_np_rs = embds_np.reshape((embds_np.shape[0]*4, int(embds_np.shape[1]/4)))
        embds_dataset = TensorDataset(torch.tensor(embds_np_rs))
        embds_dataloader = DataLoader(embds_dataset,
                                      batch_size=self.config['BATCH_SIZE'],
                                      shuffle=False)
        return embds_dataloader
