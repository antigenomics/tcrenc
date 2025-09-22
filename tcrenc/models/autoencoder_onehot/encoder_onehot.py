import pandas as pd
import numpy as np

import torch
from torch import Tensor, tensor, device
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tcrenc.models.encoder import Encoder
import tcrenc.utils.constants as constants


LEN_AA_LIST = len(constants.AA_LIST)


class Encoder_onehot(Encoder):
    def __init__(self,
                 config: dict,
                 seq_type: str):
        """
        TODO description
        """
        super(Encoder_onehot, self).__init__()

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

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def make_embeddings_from_seq(self, input_data: pd.DataFrame, device: device) -> Tensor:

        input_dataloader = self.input_data_process(inp_data=input_data[self.seq_type])

        from tcrenc.utils.run import model_process

        model_output = model_process(self,
                                     inp_dataloader=input_dataloader,
                                     device=device)

        embeddings = self.embeddings_data_process(model_output, input_data[self.seq_type])

        return embeddings

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

    def embeddings_data_process(self, encoder_output: Tensor, input_seqs: pd.Series) -> pd.DataFrame:
        """
        TODO descriprion
        """
        embeddings = encoder_output.reshape(input_seqs.shape[0], 4*self.config['LATENT_DIMS'])

        embd = pd.concat([pd.DataFrame(embeddings),
                          input_seqs.to_frame()],
                         axis=1)

        return embd

    def _make_embds_dataloader(self, input_embeddings):

        embds_np = input_embeddings.to_numpy(dtype=np.float32)
        embds_np_rs = embds_np.reshape((embds_np.shape[0]*4, int(embds_np.shape[1]/4)))
        embds_dataset = TensorDataset(tensor(embds_np_rs))
        embds_dataloader = DataLoader(embds_dataset,
                                      batch_size=self.config['BATCH_SIZE'],
                                      shuffle=False)
        return embds_dataloader

    def _embds_shape_check(self, data):
        if (data.shape[1]) != 4 * self.latent_dims:
            raise ValueError('Wrong embeddings shape')

    def model_train(self,
                    train_data: pd.DataFrame,
                    device: torch.device,
                    criterion,
                    test_data=None):

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

        from tcrenc.utils.train import part_model_train
        part_model_train(self,
                         model_type='encoder',
                         seq_train_dataloader=seq_train_dataloader,
                         embds_train_dataloader=embds_train_dataloader,
                         device=device,
                         criterion=criterion,
                         config=self.config,
                         seq_test_dataloader=seq_test_dataloader,
                         embds_test_dataloader=embds_test_dataloader)
