import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf.errors import ConfigKeyError

from tcrenc.models.autoencoder import Autoencoder
from tcrenc.models.autoencoder_onehot.encoder_onehot import Encoder_onehot
from tcrenc.models.autoencoder_onehot.decoder_onehot import Decoder_onehot
import tcrenc.utils.constants as constants
from tcrenc.utils.validate import model_validate
from tcrenc.utils.train import model_train, saving_weights


LEN_AA_LIST = len(constants.AA_LIST)


class Autoencoder_onehot(Autoencoder):
    def __init__(self,
                 config: dict,
                 seq_type: str,
                 device: torch.device):
        """
        TODO description
        """
        super(Autoencoder_onehot, self).__init__()

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

        self.encoder = Encoder_onehot(config=config,
                                      seq_type=seq_type,
                                      device=device)

        self.decoder = Decoder_onehot(config=config,
                                      seq_type=seq_type,
                                      device=device)
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def weight_load(self) -> None:
        """
        """
        self.load_state_dict(torch.load(self.weight_path,
                                        map_location=self.device,
                                        weights_only=True))

    def input_data_process(self, inp_seqs: pd.Series) -> DataLoader:
        """
        Main function to prepare torch DataLoader for input pandas Series, consist of 'cdr3' or 'antigen_epitope' sequences.
        It add gaps and ... TODO description
        """
        inp_dataloader = self.encoder.input_data_process(inp_seqs=inp_seqs)
        return inp_dataloader

    def make_embeddings_from_seq(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        """
        embeddings = self.encoder.make_embeddings_from_seq(input_data=input_data)
        return embeddings

    def make_seq_from_embeddings(self, input_embds: pd.DataFrame) -> pd.DataFrame:
        """
        """
        decoded_seqs = self.decoder.make_seq_from_embeddings(input_embds=input_embds)
        return decoded_seqs

    def reconstructed_data_process(self, input_tensor: torch.Tensor) -> pd.DataFrame:
        """
        """
        reconstructed_data = self.decoder.reconstructed_data_process(model_output=input_tensor)
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

    def save_model(self, output_path: Path) -> None:
        """
        """
        saving_weights(self, output_path, self.embd_type, self.seq_type)

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
