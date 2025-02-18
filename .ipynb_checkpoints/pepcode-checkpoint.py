import numpy as np


AA_LIST = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def one_hot_coding(peptides):
    """
    Return 3d array in all cases(included only one pep)
    """
    if type(peptides) is str:
        num_pep = 1
        len_seq = len(peptides)
    else:
        len_seq = len(peptides[0])
        num_pep = len(peptides)
        
    pep_oh_encoded = np.zeros((num_pep, len(AA_LIST), len_seq), dtype = np.float32)

    if num_pep == 1:
        for i in range(len_seq):
            for j in range(len(AA_LIST)):
                if AA_LIST[j] == peptides[i]:
                    pep_oh_encoded[0][j][i] = 1
    else:
        for k, pep in enumerate(peptides):
            for i in range(len(pep)):
                for j in range(len(AA_LIST)):
                    if AA_LIST[j] == pep[i]:
                        pep_oh_encoded[k][j][i] = 1

    return pep_oh_encoded

def one_hot_decoding(one_hot_matr):
    """
    only working with 2D matr!!!
    """
    ans = ""
    for i in range(one_hot_matr.shape[1]):
        for j in range(len(AA_LIST)):
            if one_hot_matr[j][i] == 1:
                ans += AA_LIST[j]
    return ans












        