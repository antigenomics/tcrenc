from abc import abstractmethod
import pandas as pd
from pathlib import Path

import torch
from torch.nn import Module


class Autoencoder(Module):
    @abstractmethod
    def __init__(self,
                 config: dict,
                 seq_type: str,
                 device: torch.device):
        """
        Initialize the base Autoencoder.

        Args:
            config (dict): Configuration dictionary with model parameters.
            seq_type (str): Type of sequence to process.
                Options:
                    - "antigene_epitope"
                    - "cdr3"
            device (torch.device): Target device for computation.
                Required even if the implementation does not use PyTorch.
        """
        super(Autoencoder, self).__init__()
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
                For autoencoder models there are only sequences in input.
            criterion: Loss function or evaluation criterion used for training.
            test_data (pd.DataFrame, optional): Optional DataFrame with test or validation
                data for monitoring training performance. Defaults to None.

        Returns:
            None
        """
        pass

    @abstractmethod
    def save_model(self,
                   output_path: Path):
        """
        Save the current model.

                Args:
            output_path (Path): Destination path for saving model weights.

        Returns:
            None
        """
        pass

    @abstractmethod
    def validation_on_seqs(self,
                           input_data: pd.DataFrame,
                           loss_function):
        """
        Validate the model on a set of input sequences.

        Args:
            input_data (pd.DataFrame): DataFrame containing sequences used for validation.
            loss_function: Loss function or evaluation criterion used for estimate the reconstruction error.

        Returns:
            input seqs (pd.DataFrame): DataFrame containing input sequences used for validation.
            output seqs (pd.DataFrame): DataFrame containing model ouput reconstructed sequences.
            loss_value (float): Computed loss for the validation set based on the provided loss function.
        """
        pass
