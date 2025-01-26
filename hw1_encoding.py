import peptides
import numpy as np


with open("peptides_for_test.txt") as file:
    pep = file.readline()
    pep = pep[:-1]
    print("Initial peptide: ", pep)
    AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    pep_oh_encoded = np.zeros((len(AA_LIST), len(pep)))
    for i in range(len(pep)):
        for j in range(len(AA_LIST)):
            if AA_LIST[j] == pep[i]:
                pep_oh_encoded[j][i] = 1
    print("Peptide One-Hot encoded:\n", pep_oh_encoded)
    pep1 = peptides.Peptide(pep)
    pep_af_encoded = np.zeros(5)
    for i, kf in enumerate(pep1.atchley_factors()):
        pep_af_encoded[i] = kf
    print("Peptide Atchley factors encoded:\n", pep_af_encoded)
