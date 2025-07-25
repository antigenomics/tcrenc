import pandas as pd
import numpy as np

from torch import Tensor, tensor
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tcrenc.models.autoencoder import Autoencoder
import tcrenc.utils.constants as constants


LEN_AA_LIST = len(constants.AA_LIST)


class Autoencoder_kidera(Autoencoder):
    def __init__(self,
                 config: dict,
                 seq_type: str):
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

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_dims,
                      out_features=self.latent_dims),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dims,
                      out_features=self.input_dims),
            nn.Unflatten(1, (LEN_AA_LIST, int(self.input_dims/LEN_AA_LIST))),
        )

    def forward(self, x: Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def make_embeddings_from_seq(self, x: Tensor):
        encoded = self.encoder(x)
        return encoded

    def make_seq_from_embeddings(self, encoded: Tensor):
        decoded = self.decoder(encoded)
        return decoded

    def _gap_insertion(self, inp_list: list) -> list:
        """
        Function to insert gaps to sequences of cdr3 and epitope to positions +3, +4, -3, -4
        """
        ext_list = []

        for seq in inp_list:
            gap_count = self._max_len - len(seq)

            ext_list.append(seq[0:3]+'-'*gap_count+seq[3:])
            ext_list.append(seq[0:4]+'-'*gap_count+seq[4:])
            ext_list.append(seq[0:-3]+'-'*gap_count+seq[-3:])
            ext_list.append(seq[0:-4]+'-'*gap_count+seq[-4:])

        return ext_list

    def _one_hot_code(self, peptide: str):
        """
        Return 2d np.array(np.float32): peptide in one-hot representation.
        """
        pep_oh_encoded = np.zeros((LEN_AA_LIST, len(peptide)),
                                  dtype=np.float32)

        for idx, aa in enumerate(peptide):
            aa_idx = constants.AA_LIST.index(aa)
            pep_oh_encoded[aa_idx][idx] = 1

        return pep_oh_encoded

    def input_data_process(self, inp_data: pd.Series) -> DataLoader:
        """
        Main function to prepare torch DataLoader for input pandas Series, consist of 'cdr3' or 'antigen_epitope' sequences.
        It add gaps and ... TODO description
        """
        inp_list = inp_data.to_list()
        col_name = inp_data.name

        if col_name != self.seq_type:
            raise ValueError('Processing wrong data! (CDR3 with epitope model or reverse.)')

        # Extend input list with seq's with gaps
        inp_list_with_gaps = self._gap_insertion(inp_list)

        inp_list_oh = np.zeros((len(inp_list_with_gaps),
                                LEN_AA_LIST,
                                self._max_len),
                               dtype=np.float32)

        # List of seqs with gaps to one-hot representation
        for idx, seq in enumerate(inp_list_with_gaps):
            inp_list_oh[idx] = self._one_hot_code(seq)

        inp_dataset = TensorDataset(tensor(inp_list_oh))

        inp_dataloader = DataLoader(inp_dataset,
                                    batch_size=self.config['BATCH_SIZE'],
                                    shuffle=False)
        return inp_dataloader

    def reconstructed_data_process(self, reconstructed_data: list) -> list:
        # TODO make this function
        pass
