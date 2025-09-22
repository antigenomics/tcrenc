from collections import Counter
import numpy as np
import pandas as pd

from scipy.stats import entropy
from torch import Tensor, device, tensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tcrenc.models.decoder import Decoder
import tcrenc.utils.constants as constants


LEN_AA_LIST = len(constants.AA_LIST)


class Decoder_onehot(Decoder):
    def __init__(self,
                 config: dict,
                 seq_type: str):
        """
        TODO description
        """
        super(Decoder_onehot, self).__init__()

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

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dims,
                      out_features=self.input_dims),
            nn.Unflatten(1, (LEN_AA_LIST, int(self.input_dims/LEN_AA_LIST))),
        )

    def forward(self, encoded: Tensor) -> Tensor:
        return self.decoder(encoded)

    def make_seq_from_embeddings(self, input_embds: pd.DataFrame, device: device) -> Tensor:

        input_dataloader = self.input_data_process(input_data=input_embds)

        from tcrenc.utils.run import model_process

        model_output = model_process(self,
                                     inp_dataloader=input_dataloader,
                                     device=device)

        seqs = self.reconstructed_data_process(model_output)

        return seqs, model_output

    def _one_hot_decode(self, one_hot_matr_input, mode='argmax', entropy_threshold=1):
        """
        Return peptide sequence from one-hot representation.
        Input matrix should be np array with shape (number of aminoacids, length of sequence).

        There are 2 modes to decode matrix to sequence:
        1) 'argmax'(default) - chose maximum value in the column (position in peptide) and assign it to the aminoacid.
        There is no 'X'(missing) aminoacid in output.
        2) 'entropy' - calculate the Shannon entropy of the column with scipy.stats.entropy().
        And if it more than entropy_threshold (=1 by default), then assign this position to 'X'(missing) aminoacid.
        Else find maximum in column to assign it to the aminoacid.
        """
        ans = ""
        one_hot_matr = one_hot_matr_input.copy()
        seq_len = one_hot_matr.shape[1]

        if mode == 'argmax':
            for j in range(seq_len):
                idx_max = np.argmax(one_hot_matr[:, j])
                ans += constants.AA_LIST[idx_max]
            return ans
        elif mode == 'entropy':
            for j in range(seq_len):
                if entropy(one_hot_matr[:, j]) > entropy_threshold:
                    ans += 'X'
                else:
                    idx_max = np.argmax(one_hot_matr[:, j])
                    ans += constants.AA_LIST[idx_max]
            return ans

    def _gap_removal(self, seq_output_list):

        seq_output_list_no_gap = []
        for i in range(0, len(seq_output_list), 4):

            var = []
            var.append(seq_output_list[i].replace('-', ''))
            var.append(seq_output_list[i+1].replace('-', ''))
            var.append(seq_output_list[i+2].replace('-', ''))
            var.append(seq_output_list[i+3].replace('-', ''))
            c = Counter(var)

            seq_output_list_no_gap.append(c.most_common(1)[0][0])

        return seq_output_list_no_gap

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

    def input_data_process(self, input_data: pd.DataFrame):

        embeddings = input_data.copy().to_numpy(dtype=np.float32)
        self._embds_shape_check(embeddings)

        embeddings = embeddings.reshape((embeddings.shape[0]*4, self.latent_dims))
        dataset = TensorDataset(tensor(embeddings))

        dataloader = DataLoader(dataset,
                                batch_size=self.config['BATCH_SIZE'],
                                shuffle=False)

        return dataloader

    def _embds_shape_check(self, data):
        if (data.shape[1]) != 4 * self.latent_dims:
            raise ValueError('Wrong embeddings shape')

    def reconstructed_data_process(self, model_output: Tensor):

        seq_output_list = []

        for i in range(len(model_output)):
            seq_output_list.append(self._one_hot_decode(model_output[i].numpy()))

        reconstructed_seqs = self._gap_removal(seq_output_list)
        reconstructed_seqs_df = pd.DataFrame({self.seq_type: reconstructed_seqs})

        return reconstructed_seqs_df

    def _make_seq_dataloder(self, inp_seq: pd.Series):

        inp_list = inp_seq.to_list()

        # Extend input list with seq's with gaps
        inp_list_with_gaps = self._gap_insertion(inp_list)

        inp_list_oh = np.zeros((len(inp_list_with_gaps),
                                LEN_AA_LIST,
                                self._max_len),
                               dtype=np.float32)

        # List of seqs with gaps to one-hot representation
        for idx, seq in enumerate(inp_list_with_gaps):
            inp_list_oh[idx] = self._one_hot_code(seq)

        seq_dataset = TensorDataset(tensor(inp_list_oh))

        seq_dataloader = DataLoader(seq_dataset,
                                    batch_size=self.config['BATCH_SIZE'],
                                    shuffle=False)

        return seq_dataloader

    def model_train(self,
                    train_data: pd.DataFrame,
                    device: torch.device,
                    criterion,
                    test_data=None):

        self._embds_shape_check(train_data.drop(columns=self.seq_type))

        if test_data is None:
            seq_train_dataloader = self._make_seq_dataloder(inp_seq=train_data[self.seq_type])
            seq_test_dataloader = None

            embds_train_dataloader = self.input_data_process(
                input_data=train_data.drop(columns=self.seq_type)
                )
            embds_test_dataloader = None
        else:
            seq_train_dataloader = self._make_seq_dataloder(inp_seq=train_data[self.seq_type])
            seq_test_dataloader = self._make_seq_dataloder(inp_seq=test_data[self.seq_type])

            embds_train_dataloader = self.input_data_process(
                input_data=train_data.drop(columns=self.seq_type)
                )
            embds_test_dataloader = self.input_data_process(
                input_data=test_data.drop(columns=self.seq_type)
                )

        from tcrenc.utils.train import part_model_train
        part_model_train(self,
                         model_type='decoder',
                         seq_train_dataloader=seq_train_dataloader,
                         embds_train_dataloader=embds_train_dataloader,
                         device=device,
                         criterion=criterion,
                         config=self.config,
                         seq_test_dataloader=seq_test_dataloader,
                         embds_test_dataloader=embds_test_dataloader)
