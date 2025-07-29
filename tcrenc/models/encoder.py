from abc import abstractmethod
import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module


class Encoder(Module):
    @abstractmethod
    def __init__(self,):
        super(Encoder, self).__init__()
        pass

    @abstractmethod
    def forward(self, data: Tensor) -> Tensor:
        pass

    @abstractmethod
    def input_data_process(self, inp_data: pd.Series) -> DataLoader:
        pass

    @abstractmethod
    def embeddings_data_process(self, encoder_output: Tensor, input_seqs: pd.Series) -> pd.DataFrame:
        pass
