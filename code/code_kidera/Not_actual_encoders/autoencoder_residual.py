import torch
import torch.nn as nn
class ResidualBlock(nn.Module):
    """
    A basic residual block for 2D convolutional feature maps.

    Applies two convolutional layers with batch normalization and ReLU,
    and adds the input tensor to the output (skip connection).

    Args:
        channels (int): Number of input and output channels (must match for residual connection).
    """
    def __init__(self,channels):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels,channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu=nn.ReLU()
    def forward(self,x):
        """
        Forward pass through the residual block.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Output tensor after residual connection and ReLU.
        """
        return self.relu(x+self.block(x))


class ConvAutoEncoderRes(nn.Module):
    """
    Residual Convolutional Autoencoder for encoded CDR3 amino acid sequences.

    This model is tailored for learning compact representations of CDR3 sequences
    (from T-cell) using convolutional layers with residual blocks.
    It assumes the input is a 4D tensor representing per-residue feature maps,
    e.g., Kidera factors or other amino acid embeddings.

    The encoder extracts hierarchical spatial features, compresses them into a latent
    representation, and the decoder reconstructs the original tensor with transposed convolutions.

    Args:
        linear (int): The number of amino acids in the  CDR3 sequence.
        latent_dim (int, optional): Dimension of the learned latent space. Default: 64.

    Input shape:
        (batch_size, 1, 10, linear), where:
            - 1: input channel (single channel per sample)
            - 10: number of amino acid features (e.g. Kidera factors)
            - linear: number of positions (residues) in the sequence

    Output shape:
        Same as input.
    """
    def __init__(self, linear,latent_dim=64):
        super(ConvAutoEncoderRes, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(), 
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3,  padding=1),  
            nn.BatchNorm2d(64),  
            nn.ReLU(),    
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3,  padding=1),  
            nn.BatchNorm2d(128),  
            nn.ReLU(),
            ResidualBlock(128),
            nn.Flatten()
        )
        self.linear_encode = nn.Sequential(
            nn.Linear(128 * 10*linear, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )

        self.linear_decode = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128*10*linear)
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128,10,linear)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),

            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),

            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the residual autoencoder.

        Args:
            x (Tensor): Input tensor of shape (B, 1, 10, linear)

        Returns:
            Tensor: Reconstructed tensor of same shape as input.
        """
        enc = self.encoder(x)                     
        enc = self.linear_encode(enc)             
        dec = self.linear_decode(enc)              
        out = self.decoder(dec)
        return out