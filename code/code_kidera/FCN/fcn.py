import torch.nn as nn

class SimplifiedClassifier(nn.Module):
    """
    A simple feedforward neural network classifier for binary or regression tasks.

    This model is designed to take as input a vector (latent
    embedding from an autoencoder) and predict the binding affinity
    between CDR3 and epitope sequences.

    Architecture:
        - 3 fully connected layers with:
            - Batch Normalization
            - ReLU activation
            - Dropout for regularization
        - Final linear layer projecting to output dimension 

    Args:
        input_dim (int): Dimensionality of the input vector (latent vector size).
        hidden_dim (int): Size of the first hidden layer.
        output_dim (int): Output dimension
    """
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=1):
        super(SimplifiedClassifier, self).__init__()
        
        self.fc = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.BatchNorm1d(hidden_dim // 2),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(hidden_dim // 2, hidden_dim // 4),
    nn.BatchNorm1d(hidden_dim // 4),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(hidden_dim // 4, output_dim)
)


    def forward(self, x):
        """
        Forward pass through the classifier.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.fc(x)
