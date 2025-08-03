from torch import Tensor
import torch.nn as nn

from tcrenc.models.decoder import Decoder
import tcrenc.utils.constants as constants


LEN_AA_LIST = len(constants.AA_LIST)


class Decoder_kidera(Decoder):
    def __init__(self,
                 config: dict,
                 seq_type: str):
        """
        TODO description
        """
        super(Decoder_kidera, self).__init__()

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

        self.linear_decode = nn.Sequential(
            nn.Linear(self.latent_dims, 1024), nn.ReLU(), nn.Linear(1024, 128*10*self.linear_part)
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 10, self.linear_part)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, encoded: Tensor) -> Tensor:
        return self.decoder(self.linear_decode(encoded))

    def reconstructed_data_process(self, reconstructed_data: list) -> list:
        # TODO make this function
        pass
