import pandas as pd
import os
import matplotlib as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader


def model_validate(model, input_dataloader: DataLoader, device,  criterion):
    """
    Function to create embeddings by using make_embeddings_from_seq function of model.
    It also prints reconstruction error based on criterion specified in model config file.
    """
    model.eval()

    output = []
    input = []
    loss_avg, num_batches = 0, 0

    for pre_pep in input_dataloader:
        with torch.no_grad():
            pep = pre_pep[0].to(device)
            pep_recon = model(pep)
            loss = criterion(pep_recon, pep)
            loss_avg += loss.item()
            num_batches += 1
        input.append(pep.cpu())
        output.append(pep_recon.cpu())
    loss_avg /= num_batches
    print(f'Average reconstruction error of {model.seq_type} sequences on sample: {loss_avg:.4f}')

    return torch.cat(input), torch.cat(output)


# def make_plot(err_dict: dict, err_count):

#     err_count_perc = {key/100.0: 0 for key in range(0, 100, 5)}
#     for key, val in err_count.items():
#         err_count_perc[key] = val/counter*100

#     fig, axs = plt.subplots(3, 2, figsize=[8.3, 11.7])

#     sns.barplot(dict(list(err_dict.items())[-5:]), color = default_color, ax=axs[1, 0])
#     sns.pointplot(err_count_no_gap_ce, color = default_color, ax=axs[2, 0])
#     sns.pointplot(err_count_perc_ce, color = default_color, ax=axs[2, 1])

#     axs[0, 0].set(xlabel='Epoch', ylabel='Reconstruction error', title = f'Reconstruction error during training\n', xticks=[i for i in range(1, num_epochs, int(num_epochs/5))])
#     axs[0, 1].set(frame_on=False)
#     axs[0, 1].set_xticks([])
#     axs[0, 1].set_yticks([])
#     axs[0, 1].text(x=-0.1, y=-1.5, s=f'Average reconstruction error on test set:\nCELoss: {round(test_loss_avg, 5)}\n\nFrom {len(pep_test_list_bef_no_gap)} sequences with right seq len:{counter}\n\n{classification_report(pep_test_list_bef_ae_no_gap_aa, pep_test_list_aft_ae_no_gap_aa)}')
#     axs[1, 0].set(xlabel='Aminoacid change', ylabel='Count of aminoacid change', title=f'Aminoacid changes (5 most common)')
#     axs[1, 1].set(frame_on=False)
#     axs[1, 1].set_xticks([])
#     axs[1, 1].set_yticks([])
#     axs[2, 0].set(xlabel='Relative position in peptide', ylabel='Count of errors', title=f'Count of errors by position')
#     axs[2, 1].set(xlabel='Relative position in peptide', ylabel='Percent of errors', title=f'Count of errors by position (in %)', yticks=[i*10 for i in range(1, 10)])
#     axs[2, 0].set_xticks([2,6,10,14,18])
#     axs[2, 1].set_xticks([2,6,10,14,18])

#     plt.subplots_adjust(wspace=0.3, hspace=0.5)
#     fig.suptitle(f'One-hot encoder with arch {autoencoder_arch} on test set') 
#     plt.show()
#     #fig.savefig(f'../../results/results_onehot/{str(date.today())}_{autoencoder_arch}_on_test_CELoss_final.pdf', format='pdf')

def make_report(inp_list: list, out_list: list, output_dir: str, embed_type: str, seqs_type: str) -> None:
    """
    """
    pass
