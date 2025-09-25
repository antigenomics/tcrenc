from abc import abstractmethod

from torch.nn import Module


class Autoencoder(Module):
    @abstractmethod
    def __init__(self,):
        super(Autoencoder, self).__init__()
        pass

    @abstractmethod
    def forward(self,):
        pass

    @abstractmethod
    def weight_load(self,):
        pass

    @abstractmethod
    def make_embeddings_from_seq(self,):
        pass

    @abstractmethod
    def make_seq_from_embeddings(self,):
        pass

    @abstractmethod
    def input_data_process(self,):
        pass

    @abstractmethod
    def model_train(self,):
        pass

    @abstractmethod
    def save_model(self,):
        pass

    @abstractmethod
    def validation_on_seqs(self,):
        pass
