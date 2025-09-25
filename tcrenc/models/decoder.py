from abc import abstractmethod

from torch.nn import Module


class Decoder(Module):
    @abstractmethod
    def __init__(self,):
        super(Decoder, self).__init__()
        pass

    @abstractmethod
    def forward(self,):
        pass

    @abstractmethod
    def weight_load(self,):
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
