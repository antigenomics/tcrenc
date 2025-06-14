import numpy as np
from scipy.stats import entropy
import modules.modules_onehot.constants as constants


AA_LIST = constants.AA_LIST
AA_LIST_ATCHLEY_FACTORS = constants.AA_LIST_ATCHLEY_FACTORS
BLOSUM62 = constants.BLOSUM62


def one_hot_code(peptide):
    """
    Return 2d np.array(np.float32): peptide in one-hot representation.
    """
    pep_oh_encoded = np.zeros((len(AA_LIST), len(peptide)), dtype=np.float32)

    for idx, aa in enumerate(peptide):
        aa_idx = AA_LIST.index(aa)
        pep_oh_encoded[aa_idx][idx] = 1

    return pep_oh_encoded


def one_hot_decode(one_hot_matr_input, mode='argmax', entropy_threshold=1):
    """
    Return peptide sequence from one-hot representation.
    Input matrix should be np array with shape (20(number of aminoacids), length of sequence). Values should be <=1 or it wouldn't convert it to sequence.
    There are 2 modes to decode matrix to sequence:
    1) 'armax'(default) - chose maximum value in the column (position in peptide) and assign it to the aminoacid. There is no 'X'(missing) aminoacid in output.
    2) 'entropy' - calculate the Shannon entropy of the column with scipy.stats.entropy() and if it more than entropy_threshold (=1 by default), then assign this position to 'X'(missing) aminoacid. Else find maximum in column to assign it to the aminoacid.
    """
    ans = ""
    one_hot_matr = one_hot_matr_input.copy()
    seq_len = one_hot_matr.shape[1]

    if mode == 'argmax':
        for j in range(seq_len):
            idx_max = np.argmax(one_hot_matr[:, j])
            ans += AA_LIST[idx_max]
        return ans
    elif mode == 'entropy':
        for j in range(seq_len):
            if entropy(one_hot_matr[:, j]) > entropy_threshold:
                ans += 'X'
            else:
                idx_max = np.argmax(one_hot_matr[:, j])
                ans += AA_LIST[idx_max]
        return ans


def blosum_score(a, b):
    """
    Return Blossum 62 score for 2 aminoacids.
    """
    return BLOSUM62[(''.join([a, b]))]


def atchley_factors_code(peptide):
    """
    Return 2d np.array(np.float32): peptide in representation of Atchley factors for every aminoacid.
    """
    pep_af_encoded = np.zeros((5, len(peptide)), dtype=np.float32)

    for i, k in enumerate(peptide):
        for j in range(5):
            pep_af_encoded[j][i] = AA_LIST_ATCHLEY_FACTORS[k][j]

    return pep_af_encoded


def atchley_factors_decode(pep_af_matr):
    """
    Return peptide sequence from Atchley factors representation. The nearest aminoacid choose.
    """
    ans = ""

    for i in range(pep_af_matr.shape[1]):
        min_value = float("inf")
        key_min = ''
        for j in AA_LIST_ATCHLEY_FACTORS.keys():
            if ((pep_af_matr[:, i] - AA_LIST_ATCHLEY_FACTORS[j])**2).sum() < min_value:
                min_value = ((pep_af_matr[:, i] - AA_LIST_ATCHLEY_FACTORS[j]) ** 2).sum()
                key_min = j
        ans += key_min

    return ans
