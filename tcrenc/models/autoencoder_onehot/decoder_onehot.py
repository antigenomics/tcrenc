from torch import Tensor
import torch.nn as nn

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

    def reconstructed_data_process(self, reconstructed_data: list) -> list:
        # TODO make this function
        pass
