from abc import abstractmethod
import pandas as pd

import torch
from torch.nn import Module


class Decoder(Module):
    @abstractmethod
    def __init__(self,
                 config: dict,
                 seq_type: str,
                 device: torch.device):
        super(Decoder, self).__init__()
        """
        Initialize the base Decoder.

        Args:
            config (dict): Configuration dictionary with model parameters.
            seq_type (str): Type of sequence to process.
                Options:
                    - "antigene_epitope"
                    - "cdr3"
            device (torch.device): Target device for computation.
                Required even if the implementation does not use PyTorch.
        """
        pass

    @abstractmethod
    def weight_load(self):
        """
        Abstract method for loading model weights.

        For non-NN models, this method can be left empty but must still be implemented.
        """
        pass

    @abstractmethod
    def make_seq_from_embeddings(self,
                                 input_embds: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct sequences from input embeddings.

        Args:
            input_embds (pd.DataFrame): A Pandas DataFrame containing embeddings.
                Each row should represent a single sequence embedding.

        Returns:
            pd.DataFrame: A DataFrame containing reconstructed sequences.
        """
        pass

    @abstractmethod
    def model_train(self,
                    train_data: pd.DataFrame,
                    criterion,
                    test_data: pd.DataFrame = None) -> None:
        """
        Train the model using the provided training data.

        Args:
            train_data (pd.DataFrame): DataFrame containing input training data.
                For decoder models there are sequences and embeddings in input.
            criterion: Loss function or evaluation criterion used for training.
            test_data (pd.DataFrame, optional): Optional DataFrame with test or validation
                data for monitoring training performance. Defaults to None.

        Returns:
            None
        """
        pass

    @abstractmethod
    def save_model(self):
        """
        Save the current model.

        Returns:
            None
        """
        pass
