import torch.nn as nn
from modules.autoencoder import Autoencoder
import modules.params as params


latent_dims = params.LATENT_DIMS


class Autoencoder_onehot(Autoencoder):
    def __init__(self, input_dims):
        super(Autoencoder_onehot, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dims, out_features=latent_dims),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dims, out_features=input_dims),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def make_embeddings_from_seq(self, x):
        encoded = self.encoder(x)
        return encoded

    def make_seq_from_embeddings(self, encoded):
        decoded = self.decoder(encoded)
        return decoded
