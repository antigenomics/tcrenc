from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf.errors import ConfigKeyError

from tcrenc.models.decoder import Decoder
import tcrenc.utils.constants as constants
from tcrenc.utils.train import part_model_train, saving_weights
from tcrenc.utils.run import model_process


LEN_AA_LIST = len(constants.AA_LIST)


class Decoder_onehot(Decoder):
    """
    One-hot Decoder model.

    This model construct protein sequences('cdr3' or 'antigene_epitope') from latent space representations
    using one-hot encoding.

    See `Decoder` for the base methods description.
    """
    def __init__(self,
                 config: dict,
                 seq_type: str,
                 device: torch.device):
        """
        Initializes the one-hot decoder.

        Args:
            config: Configuration dictionary containing model parameters
            seq_type: Type of sequence ('cdr3' or 'antigen_epitope')
            device: Torch device for computation (CPU/GPU)

        Raises:
            ValueError: If unknown sequence type is provided
            ConfigKeyError: If weight path is not found in config
        """
        super(Decoder_onehot, self).__init__()

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

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dims,
                      out_features=self.input_dims),
            nn.Unflatten(1, (LEN_AA_LIST, int(self.input_dims/LEN_AA_LIST))),
        )

        self.device = device
        self.to(device)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder network.

        Args:
            encoded: Input tensor of shape (batch_size, latent_dims)

        Returns:
            Decoded tensor of shape (batch_size, LEN_AA_LIST, seq_length)
        """
        return self.decoder(encoded)

    def weight_load(self) -> None:
        """
        Loads pretrained weights for the model.
        """
        self.load_state_dict(torch.load(self.weight_path,
                                        map_location=self.device,
                                        weights_only=True))

    def input_data_process(self, input_embds: pd.DataFrame) -> DataLoader:
        """
        Prepare a DataLoader from a Pandas DataFrame of embeddings.

        Args:
            input_embds: DataFrame containing embeddings of shape (n_samples, 4*latent_dims)

        Returns:
            DataLoader with processed embeddings

        Raises:
            ValueError: If input embeddings have incorrect shape
        """
        embeddings = input_embds.copy().to_numpy(dtype=np.float32)
        self._embds_shape_check(embeddings)

        embeddings = embeddings.reshape((embeddings.shape[0]*4, self.latent_dims))

        dataset = TensorDataset(torch.tensor(embeddings))
        dataloader = DataLoader(dataset,
                                batch_size=self.config['BATCH_SIZE'],
                                shuffle=False)
        return dataloader

    def make_seq_from_embeddings(self, input_embds: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct sequences from input embeddings.

        Args:
            input_embds: DataFrame containing embeddings to decode

        Returns:
            DataFrame with decoded sequences
        """
        input_dataloader = self.input_data_process(input_embds=input_embds)

        model_output = model_process(self,
                                     inp_dataloader=input_dataloader,
                                     device=self.device)

        seqs = self.reconstructed_data_process(model_output)

        return seqs

    def reconstructed_data_process(self, model_output: torch.Tensor) -> pd.DataFrame:
        """
        Post-processes model output into final sequences.

        Args:
            model_output: Tensor of shape (batch_size, LEN_AA_LIST, seq_length)

        Returns:
            DataFrame containing decoded sequences
        """
        seq_output_list = []

        for i in range(len(model_output)):
            seq_output_list.append(self._one_hot_decode(model_output[i].numpy()))

        reconstructed_seqs = self._gap_removal(seq_output_list)
        reconstructed_seqs_df = pd.DataFrame({self.seq_type: reconstructed_seqs})

        return reconstructed_seqs_df

    def model_train(self,
                    train_data: pd.DataFrame,
                    criterion,
                    test_data=None) -> None:
        """
        Trains the decoder model.

        Args:
            train_data: DataFrame containing training sequences and embeddings
            criterion: Loss function for training
            test_data: Optional DataFrame for validation data

        Raises:
            ValueError: If input data has incorrect shape
        """
        self._embds_shape_check(train_data.drop(columns=self.seq_type))

        if test_data is None:
            seq_train_dataloader = self._make_seq_dataloder(inp_seq=train_data[self.seq_type])
            seq_test_dataloader = None

            embds_train_dataloader = self.input_data_process(
                input_embds=train_data.drop(columns=self.seq_type)
                )
            embds_test_dataloader = None
        else:
            seq_train_dataloader = self._make_seq_dataloder(inp_seq=train_data[self.seq_type])
            seq_test_dataloader = self._make_seq_dataloder(inp_seq=test_data[self.seq_type])

            embds_train_dataloader = self.input_data_process(
                input_embds=train_data.drop(columns=self.seq_type)
                )
            embds_test_dataloader = self.input_data_process(
                input_embds=test_data.drop(columns=self.seq_type)
                )

        part_model_train(self,
                         model_type='decoder',
                         seq_train_dataloader=seq_train_dataloader,
                         embds_train_dataloader=embds_train_dataloader,
                         device=self.device,
                         criterion=criterion,
                         config=self.config,
                         seq_test_dataloader=seq_test_dataloader,
                         embds_test_dataloader=embds_test_dataloader)

    def validation_on_seqs(self, input_data: pd.DataFrame, loss_function):
        """
        Validates model performance on input sequences.

        Args:
            input_data: DataFrame containing input sequences and embeddings
            loss_function: Function to compute validation loss

        Returns:
            tuple: (input_seqs, output_seqs, loss_value)
            input seqs (pd.DataFrame): DataFrame containing input sequences used for validation.
            output seqs (pd.DataFrame): DataFrame containing model ouput reconstructed sequences.
            loss_value (float): Computed loss for the validation set based on the provided loss function.
        """

        input_seqs = input_data[self.seq_type].to_frame()
        embds = input_data.copy().drop(columns=self.seq_type)
        output_seqs = self.make_seq_from_embeddings(input_embds=embds)

        inp_seq_dataloader = self._make_seq_dataloder(inp_seq=input_seqs[self.seq_type])
        out_seq_dataloader = self._make_seq_dataloder(inp_seq=output_seqs[self.seq_type])

        loss_value, num_batches = 0, 0
        for (inp_seq, out_seq) in zip(inp_seq_dataloader, out_seq_dataloader):
            loss = loss_function(inp_seq, out_seq)
            loss_value += loss.item()
            num_batches += 1
        loss_value /= num_batches

        return input_seqs, output_seqs, loss_value

    def save_model(self, output_path: Path) -> None:
        """
        Saves model weights to specified path.

        Args:
            output_path: Path to save model weights
        """
        saving_weights(self, output_path, self.embd_type, self.seq_type)

    def _one_hot_decode(self, one_hot_matr_input: np.ndarray) -> str:
        """
        Decodes one-hot matrix into amino acid sequence.

        Args:
            one_hot_matr_input: 2D numpy array of shape (LEN_AA_LIST, seq_length)

        Returns:
            Decoded amino acid sequence string
        """
        ans = ""
        one_hot_matr = one_hot_matr_input.copy()
        seq_len = one_hot_matr.shape[1]

        for j in range(seq_len):
            idx_max = np.argmax(one_hot_matr[:, j])
            ans += constants.AA_LIST[idx_max]
        return ans

    def _gap_removal(self, seq_output_list: list) -> list:
        """
        Removes gaps from sequences made by this model and selects most common variant.

        Args:
            seq_output_list: List of sequences with gaps

        Returns:
            List of sequences without gaps

        Note:
            This function works with 4 sequences made from one representation in latent space.
        """
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
            ValueError: If embeddings have incorrect shape
        """
        if (embds_inp_data.shape[1]) != 4 * self.latent_dims:
            raise ValueError('Wrong embeddings shape')

    def _make_seq_dataloder(self, inp_seq: pd.Series) -> DataLoader:
        """
        Creates DataLoader from input sequences.

        Args:
            inp_seq: Pandas Series containing sequences

        Returns:
            DataLoader with one-hot encoded sequences
        """
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

        seq_dataset = TensorDataset(torch.tensor(inp_list_oh))

        seq_dataloader = DataLoader(seq_dataset,
                                    batch_size=self.config['BATCH_SIZE'],
                                    shuffle=False)
        return seq_dataloader
