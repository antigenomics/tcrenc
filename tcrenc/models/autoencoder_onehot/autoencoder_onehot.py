import pandas as pd

import torch
from torch.utils.data import DataLoader

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
                 seq_type: str):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def weight_load(self, weight_path: str, device: torch.device):

        self.load_state_dict(torch.load(weight_path,
                                        map_location=device,
                                        weights_only=True))

    def input_data_process(self, inp_data: pd.Series) -> DataLoader:
        """
        Main function to prepare torch DataLoader for input pandas Series, consist of 'cdr3' or 'antigen_epitope' sequences.
        It add gaps and ... TODO description
        """
        inp_dataloader = self.encoder.input_data_process(inp_data=inp_data)
        return inp_dataloader

    def make_embeddings_from_seq(self, input_data: pd.DataFrame, device: torch.device) -> torch.Tensor:
        embeddings = self.encoder.make_embeddings_from_seq(input_data=input_data, device=device)
        return embeddings

    def embeddings_data_process(self, encoder_output: torch.Tensor, input_seqs: pd.Series) -> pd.DataFrame:
        """
        """
        embd = self.encoder.embeddings_data_process(encoder_output=encoder_output,
                                                    input_seqs=input_seqs)

        return embd

    def make_seq_from_embeddings(self, input_embds: torch.Tensor, device: torch.device) -> torch.Tensor:
        decoded_seqs, decoded = self.decoder.make_seq_from_embeddings(input_embds=input_embds, device=device)
        return decoded_seqs, decoded

    def reconstructed_data_process(self, input_tensor: torch.Tensor):
        """
        """
        reconstructed_data = self.decoder.reconstructed_data_process(model_output=input_tensor)
        return reconstructed_data

    def model_train(self,
                    input_dataloader: DataLoader,
                    device: torch.device,
                    criterion,
                    input_train_seqs,
                    test_dataloader=None):

        model_train(self,
                    input_dataloader=input_dataloader,
                    device=device,
                    criterion=criterion,
                    config=self.config,
                    test_dataloader=test_dataloader)

    def save_model(self, output_path):
        saving_weights(self, output_path, self.embd_type, self.seq_type)

    def validation_on_seqs(self, input_data: pd.DataFrame, loss_function, device):

        input_dataloader = self.input_data_process(inp_data=input_data[self.seq_type])

        _, output_seqs_coded, loss_value = model_validate(self,
                                                          input_dataloader=input_dataloader,
                                                          device=device,
                                                          criterion=loss_function)

        input_seqs = input_data[self.seq_type].to_frame()
        output_seqs = self.reconstructed_data_process(output_seqs_coded)

        return input_seqs, output_seqs, loss_value

    def encoder_part(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return encoded

    def decoder_part(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(x)
        return decoded
