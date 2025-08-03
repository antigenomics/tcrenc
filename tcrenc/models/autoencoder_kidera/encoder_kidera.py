import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tcrenc.models.encoder import Encoder
import tcrenc.utils.constants as constants


LEN_AA_LIST = len(constants.AA_LIST)
KIDERA_DICT = constants.AA_LIST_KIDERA_FACTORS_scaled


class Encoder_kidera(Encoder):
    def __init__(self,
                 config: dict,
                 seq_type: str):
        """
        TODO description
        """
        super(Encoder_kidera, self).__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_encode(self.encoder(x))

    def _gap_insertion(self, inp_seq: str) -> str:

        """
        Insert gaps (-) to make all sequences with max length
        Args:
            inp_seq: Amino acid sequence
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
        data_tensor = torch.tensor(
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

    def embeddings_data_process(self, encoder_output: torch.Tensor, input_seqs: pd.Series) -> pd.DataFrame:
        """
        TODO descriprion
        """

        embd = pd.concat([pd.DataFrame(encoder_output),
                          input_seqs.to_frame()],
                         axis=1)

        return embd
