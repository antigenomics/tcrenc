import torch
import torch.nn as nn

class ConvAutoEncoder(nn.Module):
    """
    Convolutional Autoencoder for encoded CDR3 amino acid sequences.

    This model is designed to learn compact latent representations of CDR3 sequences
    by using convolutional and fully connected layers.
    It operates on numerical feature representations of amino acids (kidera factors) shaped as 2D tensors.

    Args:
        linear (int): The length of the CDR3 sequence (or adjusted width after encoding),
                      which determines the spatial dimension of the convolutional input.
        latent_dim (int, optional): Size of the latent space vector. Default is 64.

    Architecture:
        - Encoder:
            3 convolutional layers with batch normalization and ReLU, followed by flattening.
        - Latent Projection:
            A bottleneck of fully connected layers maps the high-dimensional features into
            a lower-dimensional latent space.
        - Decoder:
            The latent vector is reconstructed through linear layers, reshaped, and passed
            through transposed convolutions to reconstruct the input.

    Note:
        - Input tensor shape must be (batch_size, 1, 10, linear), where `10` is the number of
          amino acid features per residue, and `linear` is the number of
          amino acids in the CDR3 sequence (possibly padded to a fixed length).
    """
    def __init__(self, linear, latent_dim=64):
        """
        Initializes the convolutional autoencoder.

        Args:
            linear (int): Number of amino acid positions in the CDR3 sequence.
            latent_dim (int): Dimension of the latent embedding space.
        """
        super(ConvAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),  
            nn.ReLU(),    
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),  
            nn.ReLU(),
            nn.Flatten()
        )
        self.linear_encode = nn.Sequential(
            nn.Linear(128 * 10 * linear, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )

        self.linear_decode = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 10 * linear)
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 10, linear)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 10, linear),
                              where 10 is the number of features per amino acid,
                              and linear is the number of amino acids.

        Returns:
            torch.Tensor: Reconstructed input tensor of the same shape.
        """
        enc = self.encoder(x)                      
        enc = self.linear_encode(enc)              
        dec = self.linear_decode(enc)              
        out = self.decoder(dec)
        return out
