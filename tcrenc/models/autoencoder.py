from abc import abstractmethod
import pandas as pd
from torch.nn import Module
from torch.utils.data import DataLoader


class Autoencoder(Module):
    @abstractmethod
    def __init__(self,):
        super(Autoencoder, self).__init__()
        pass

    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def make_embeddings_from_seq(self, data):
        pass

    @abstractmethod
    def make_seq_from_embeddings(self, data):
        pass

    @abstractmethod
    def input_data_process(self, inp_data: pd.Series) -> DataLoader:
        pass

    @abstractmethod
    def reconstructed_data_process(self, reconstructed_data: list) -> list:
        pass

    # @abstractmethod need this? May be here something like read config and fc init
    # def params_process(self, data):
    #     pass
