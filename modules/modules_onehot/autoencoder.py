import torch.nn as nn
import modules.modules_onehot.constants as constants


latent_dims = constants.LATENT_DIMS


class Autoencoder(nn.Module):
    def __init__(self, input_dims):
        super(Autoencoder, self).__init__()
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

    def encoding(self, x):
        encoded = self.encoder(x)
        return encoded

    def decoding(self, encoded):
        decoded = self.decoder(encoded)
        return decoded
