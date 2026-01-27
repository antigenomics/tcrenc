from abc import abstractmethod
import pandas as pd

import torch
from torch.nn import Module


class Encoder(Module):
    @abstractmethod
    def __init__(self,
                 config: dict,
                 seq_type: str,
                 device: torch.device):
        """
        Initialize the base Encoder.

        Args:
            config (dict): Configuration dictionary with model parameters.
            seq_type (str): Type of sequence to process.
                Options:
                    - "antigene_epitope"
                    - "cdr3"
            device (torch.device): Target device for computation.
                Required even if the implementation does not use PyTorch.
        """
        super(Encoder, self).__init__()
        pass

    @abstractmethod
    def weight_load(self):
        """
        Abstract method for loading model weights.

        For non-NN models, this method can be left empty but must still be implemented.
        """
        pass

    @abstractmethod
    def make_embeddings_from_seq(self,
                                 input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create embeddings from input biological sequences.

        Args:
            input_data (pd.DataFrame): A Pandas DataFrame containing a single column
                with one type of sequences (either `cdr3` or `antigen_epitope`).

        Returns:
            pd.DataFrame: A DataFrame containing generated embeddings.
                The original input sequences are included in the last column.
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
                For encoder models there are sequences and embeddings in input.
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
