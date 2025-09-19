import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from collections import defaultdict

import torch
from torch.utils.data import DataLoader


sns.set_theme(style="darkgrid")
default_color = "#6193d8"


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

    return torch.cat(input), torch.cat(output), loss_avg


def make_report(input_seqs: pd.DataFrame,
                output_seqs: pd.DataFrame,
                output_dir,
                embed_type: str,
                seq_type: str,
                loss_value: float
                ):

    report_dict = {}

    input_len_dstrb = input_seqs[seq_type].str.len().value_counts().to_dict()
    output_len_dstrb = output_seqs[seq_type].str.len().value_counts().to_dict()

    rel_position_lst = np.round(np.linspace(0.0, 1.0, num=21), 2)
    err_position_dict = {float(key): 0 for key in rel_position_lst.tolist()}

    err_dict = {}
    err_df = pd.DataFrame(columns=['Sequence_before', 'Sequence_after'])
    err_df_idx = 0

    new_seq_len_dict = defaultdict(int)
    new_seq_len_df = pd.DataFrame(columns=['Sequence_before', 'Sequence_after'])
    new_seq_len_df_idx = 0

    right_len_counter = 0
    right_seq_counter = 0

    for seq_bef, seq_aft in zip(input_seqs[seq_type].values, output_seqs[seq_type].values):

        len_bef = len(seq_bef)
        len_aft = len(seq_aft)

        if len_bef != len_aft:
            new_seq_len_dict[f'{len_bef}'+f'-{len_aft}'] += 1
            new_seq_len_df.loc[new_seq_len_df_idx] = [seq_bef, seq_aft]
            new_seq_len_df_idx += 1
            continue

        right_len_counter += 1

        flag = 0
        for pos, (aa_bef, aa_aft) in enumerate(zip(seq_bef, seq_aft)):

            if aa_bef != aa_aft:

                err_rel_pos = round(pos/len(seq_bef), 2)
                idx = (np.abs(rel_position_lst - err_rel_pos)).argmin()
                target_pos = rel_position_lst[idx]

                err_position_dict[target_pos] += 1

                if aa_bef+aa_aft in err_dict.keys():
                    err_dict[aa_bef+aa_aft] += 1
                else:
                    err_dict[aa_bef+aa_aft] = 1

                flag = 1

        if flag == 1:
            err_df.loc[err_df_idx] = [seq_bef, seq_aft]
            err_df_idx += 1
        else:
            right_seq_counter += 1

    err_dict = dict(sorted(err_dict.items(), key=lambda item: item[1]))

    report_dict['GENERAL'] = {'LOSS_VALUE': loss_value}

    report_dict['AMINO_ACID_CHANGES'] = {'POSITIONS': dict(err_position_dict),
                                         'CHANGES': err_dict}

    report_dict['SEQUENCE_LENGHT'] = {'BEFORE': input_len_dstrb,
                                      'AFTER': output_len_dstrb,
                                      'CHANGES': dict(new_seq_len_dict)
                                      }

    output_report_path = f'{output_dir}/main_report_{seq_type}_{embed_type}.yaml'
    output_err_df_path = f'{output_dir}/wrong_sequences_right_len_{seq_type}_{embed_type}.csv'
    output_new_seq_len_df_path = f'{output_dir}/wrong_sequences_wrong_len_{seq_type}_{embed_type}.csv'

    new_seq_len_df.to_csv(output_new_seq_len_df_path)
    err_df.to_csv(output_err_df_path)

    with open(output_report_path, 'w+') as f:
        yaml.dump(report_dict, f)

    err_position_dict_perc = {float(key): 0 for key in rel_position_lst.tolist()}
    for key, val in err_position_dict.items():
        err_position_dict_perc[key] = val/right_len_counter*100

    fig, axs = plt.subplots(2, 2, figsize=[8.3, 8.3])
    sns.barplot(dict(list(err_dict.items())[-5:]),
                color=default_color,
                ax=axs[0, 0],
                linewidth=0.5, edgecolor=".1")

    sns.barplot(err_position_dict,
                color=default_color,
                ax=axs[1, 0],
                linewidth=0.5, edgecolor=".1")

    sns.barplot(err_position_dict_perc,
                color=default_color,
                ax=axs[1, 1],
                linewidth=0.5, edgecolor=".1")

    axs[0, 0].set(xlabel='Aminoacid change',
                  ylabel='Rate of changes',
                  title='Aminoacid changes (5 most common)')
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    axs[0, 1].set(frame_on=False)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    plot_report = (
        f'Average reconstruction error (Loss): {round(loss_value, 4)}\n\n'
        f'From {input_seqs.shape[0]}: sequences with right length - {right_len_counter}\n'
        f'right sequences - {right_seq_counter}'
    )

    axs[0, 1].text(x=0, y=0.6, s=plot_report)

    axs[1, 0].set(xlabel='Relative position in sequence',
                  ylabel='Rate of errors',
                  title='Classification error rate by positions\n in a sequence',
                  xticks=[2, 6, 10, 14, 18])
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    axs[1, 1].set(xlabel='Relative position in sequence',
                  ylabel='Rate of errors (%)',
                  title='Classification error rate by positions\n in a sequence(%)',
                  xticks=[2, 6, 10, 14, 18])
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle(f'Validation report for autoencoder for {seq_type} sequences with {embed_type} representation')
    plt.tight_layout()

    output_plot_path = f'{output_dir}/report_plot_{seq_type}_{embed_type}.pdf'
    fig.savefig(output_plot_path,  format='pdf')
