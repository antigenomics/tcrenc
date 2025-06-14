import torch
def GPU(use_gpu):
    '''
        This function allows you using GPU systems. If you have MAC M1 you will use mps system.
        Arguments:
            - use_gpu (bool) - True if you want to use GPU.
    '''
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    elif use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")