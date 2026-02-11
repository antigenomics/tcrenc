import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf.errors import ConfigKeyError

from tcrenc.models.autoencoder import Autoencoder
from tcrenc.models.autoencoder_onehot.encoder_onehot import Encoder_onehot
from tcrenc.models.autoencoder_onehot.decoder_onehot import Decoder_onehot
import tcrenc.utils.constants as constants
from tcrenc.utils.validate import model_validate
from tcrenc.utils.train import model_train, saving_weights


LEN_AA_LIST = len(constants.AA_LIST)


class Autoencoder_onehot(Autoencoder):
    """
    One-hot autoencoder model.

    This model composes an `Encoder_onehot` and `Decoder_onehot` and follows the
    `Autoencoder` interface.

    See `Autoencoder` for the base methods description.
    """

    def __init__(self,
                 config: dict,
                 seq_type: str,
                 device: torch.device):
        """
        Initialize the one-hot-based autoencoder.

        Args:
            config: Configuration dictionary containing model parameters
            seq_type: Type of sequence ('cdr3' or 'antigen_epitope')
            device: Torch device for computation (CPU/GPU)

        Raises:
            ValueError: If unknown sequence type is provided
            ConfigKeyError: If weight path is not found in config
        """
        super(Autoencoder_onehot, self).__init__()

        self.config = config
        self.seq_type = seq_type
        self.embd_type = 'onehot'

        if self.seq_type == 'cdr3':
            self._max_len = self.config['MAX_CDR3_LEN']
            self.input_dims = self._max_len * LEN_AA_LIST
            try:
                self.weight_path = self.config['WEIGHTS_CDR3']
            except ConfigKeyError:
                self.weight_path = None

        elif self.seq_type == 'antigen_epitope':
            self._max_len = self.config['MAX_EPITOPE_LEN']
            self.input_dims = self._max_len * LEN_AA_LIST
            try:
                self.weight_path = self.config['WEIGHTS_EPIOPE']
            except ConfigKeyError:
                self.weight_path = None

        else:
            raise ValueError('Unknown seq type for this model.')

        self.latent_dims = self.config['LATENT_DIMS']

        self.encoder = Encoder_onehot(config=config,
                                      seq_type=seq_type,
                                      device=device)

        self.decoder = Decoder_onehot(config=config,
                                      seq_type=seq_type,
                                      device=device)
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder model.

        Args:
            x (torch.Tensor): Input tensor to the autoencoder.

        Returns:
            torch.Tensor: Reconstructed tensor output by the decoder.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def weight_load(self) -> None:
        """
        Load model weights from the configured path.

        Returns:
            None
        """
        self.load_state_dict(torch.load(self.weight_path,
                                        map_location=self.device,
                                        weights_only=True))

    def input_data_process(self, inp_seqs: pd.Series) -> DataLoader:
        """
        Prepare a DataLoader from a Pandas Series of sequences.

        Delegates to the encoder preprocessing pipeline (e.g., padding
        and one-hot encoding) and wraps the data into a torch DataLoader.

        Args:
            inp_seqs (pd.Series): Series of input sequences matching the configured
                `seq_type`.

        Returns:
            torch.utils.data.DataLoader: DataLoader ready for training, validation or making embeddings.
        """
        inp_dataloader = self.encoder.input_data_process(inp_seqs=inp_seqs)
        return inp_dataloader

    def make_embeddings_from_seq(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create embeddings from input biological sequences.
        Delegates to the encoder same method.

        Args:
            input_data (pd.DataFrame): A DataFrame containing a single column
                with one type of sequences.

        Returns:
            pd.DataFrame: A DataFrame containing generated embeddings.
                The original input sequences are included in the last column.
        """
        embeddings = self.encoder.make_embeddings_from_seq(input_data=input_data)
        return embeddings

    def make_seq_from_embeddings(self, input_embds: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct sequences from input embeddings.
        Delegates to the decoder same method.

        Args:
            input_embds (pd.DataFrame): A DataFrame containing embeddings where
                each row represents a single sequence embedding.

        Returns:
            pd.DataFrame: A DataFrame containing reconstructed sequences.
        """
        decoded_seqs = self.decoder.make_seq_from_embeddings(input_embds=input_embds)
        return decoded_seqs

    def reconstructed_data_process(self, input_tensor: torch.Tensor) -> pd.DataFrame:
        """
        Post-process decoder output into a DataFrame with reconstructed sequences.

        Delegates to `decoder.reconstructed_data_process`.

        Args:
            input_tensor (torch.Tensor): Model output tensor.

        Returns:
            pd.DataFrame: DataFrame containing reconstructed sequences.
        """
        reconstructed_data = self.decoder.reconstructed_data_process(model_output=input_tensor)
        return reconstructed_data

    def model_train(self,
                    train_data: pd.DataFrame,
                    criterion,
                    test_data=None) -> None:
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

        input_dataloader = self.input_data_process(inp_seqs=train_data[self.seq_type])

        if test_data is not None:
            test_dataloader = self.input_data_process(inp_seqs=test_data[self.seq_type])
        else:
            test_dataloader = test_data

        model_train(self,
                    input_dataloader=input_dataloader,
                    device=self.device,
                    criterion=criterion,
                    config=self.config,
                    test_dataloader=test_dataloader)

    def save_model(self, output_path: Path) -> None:
        """
        Save the current model.

        Args:
            output_path (Path): Destination path for saving model weights.

        Returns:
            None
        """
        saving_weights(self, output_path, self.embd_type, self.seq_type)

    def validation_on_seqs(self, input_data: pd.DataFrame, loss_function):
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
        input_dataloader = self.input_data_process(inp_seqs=input_data[self.seq_type])

        _, output_seqs_coded, loss_value = model_validate(self,
                                                          input_dataloader=input_dataloader,
                                                          device=self.device,
                                                          criterion=loss_function)

        input_seqs = input_data[self.seq_type].to_frame()
        output_seqs = self.reconstructed_data_process(output_seqs_coded)

        return input_seqs, output_seqs, loss_value
