from torch.nn import Module
from abc import abstractmethod


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
