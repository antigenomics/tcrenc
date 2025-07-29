from abc import abstractmethod
from torch.nn import Module
from torch import Tensor


class Decoder(Module):
    @abstractmethod
    def __init__(self,):
        super(Decoder, self).__init__()
        pass

    @abstractmethod
    def forward(self, data: Tensor) -> Tensor:
        pass

    @abstractmethod
    def reconstructed_data_process(self, reconstructed_data: Tensor) -> list:
        pass
