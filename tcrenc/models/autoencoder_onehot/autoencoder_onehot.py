import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from tcrenc.models.autoencoder import Autoencoder
import tcrenc.utils.constants as constants
from tcrenc.models.autoencoder_onehot.encoder_onehot import Encoder_onehot
from tcrenc.models.autoencoder_onehot.decoder_onehot import Decoder_onehot


LEN_AA_LIST = len(constants.AA_LIST)


class Autoencoder_onehot(Autoencoder):
    def __init__(self,
                 config: dict,
                 seq_type: str):
        """
        TODO description
        """
        super(Autoencoder_onehot, self).__init__()

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

        self.encoder = Encoder_onehot(config=config,
                                      seq_type=seq_type)

        self.decoder = Decoder_onehot(config=config,
                                      seq_type=seq_type)

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def make_embeddings_from_seq(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        return encoded

    def make_seq_from_embeddings(self, encoded: Tensor) -> Tensor:
        decoded = self.decoder(encoded)
        return decoded

    def model_process(self, input_dataloader: DataLoader, device: torch.device, criterion, process_type: str) -> Tensor:

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

    def embeddings_data_process(self, encoder_output: Tensor, input_seqs: pd.Series) -> pd.DataFrame:
        """
        TODO descriprion
        """
        embd = self.encoder.embeddings_data_process(encoder_output=encoder_output,
                                                    input_seqs=input_seqs)

        return embd
