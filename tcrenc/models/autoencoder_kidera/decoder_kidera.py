import pandas as pd
import numpy as np
import pickle
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from tcrenc.models.decoder import Decoder
import tcrenc.utils.constants as constants
from tcrenc.utils.train import part_model_train, saving_weights
from tcrenc.utils.run import model_process

LEN_AA_LIST = len(constants.AA_LIST)
KIDERA_DICT = constants.AA_LIST_KIDERA_FACTORS_scaled


class Decoder_kidera(Decoder):
    def __init__(self,
                 config: dict,
                 seq_type: str,
                 device: torch.device):
        """
        TODO description
        """
        super(Decoder_kidera, self).__init__()

        self.config = config
        self.seq_type = seq_type
        self.embd_type = 'kidera'

        if self.seq_type == 'cdr3':
            self._max_len = self.config['MAX_CDR3_LEN']
            self.input_dims = self._max_len * LEN_AA_LIST
            self.linear_part = self.config['LINEAR_PART_CDR3']
        elif self.seq_type == 'antigen_epitope':
            self._max_len = self.config['MAX_EPITOPE_LEN']
            self.input_dims = self._max_len * LEN_AA_LIST
            self.linear_part = self.config['LINEAR_PART_EPITOPE']
        else:
            raise ValueError('Unknown seq type for this model.')

        self.latent_dims = self.config['LATENT_DIMS']

        self.linear_decode = nn.Sequential(
            nn.Linear(self.latent_dims, 1024), nn.ReLU(), nn.Linear(1024, 128*10*self.linear_part)
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 10, self.linear_part)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.umap = None
        self.rfc = None

        self.to(device)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.linear_decode(encoded))

    def weight_load(self, weight_path: str, device: torch.device):

        self.load_state_dict(torch.load(weight_path,
                                        map_location=device,
                                        weights_only=True))

        # Make path to umap
        directory = os.path.dirname(weight_path)

        umap_filename = f'umap_{self.embd_type}_{self.seq_type}.pkl'
        umap_path = os.path.join(directory, umap_filename)

        with open(umap_path, "rb") as f:
            self.umap = pickle.load(f)

        rfc_filename = f'rfc_{self.embd_type}_{self.seq_type}.pkl'
        rfc_path = os.path.join(directory, rfc_filename)
        with open(rfc_path, "rb") as f:
            self.rfc = pickle.load(f)

    def input_data_process(self, input_data: pd.DataFrame):

        embeddings = input_data.copy().to_numpy(dtype=np.float32)
        self._embds_shape_check(embeddings)

        dataset = TensorDataset(torch.tensor(embeddings))
        dataloader = DataLoader(dataset,
                                batch_size=self.config['BATCH_SIZE'],
                                shuffle=False)

        return dataloader

    def make_seq_from_embeddings(self, input_embds: pd.DataFrame,
                                 device: torch.device) -> torch.Tensor:

        input_dataloader = self.input_data_process(input_data=input_embds)

        model_output = model_process(self,
                                     inp_dataloader=input_dataloader,
                                     device=device)

        seqs = self.reconstructed_data_process(model_output)

        return seqs, model_output

    def reconstructed_data_process(self, reconstructed_data):

        def_dict = {i: x for i, x in enumerate(KIDERA_DICT.keys())}

        reshaped_data = []
        for seq_tensor in reconstructed_data:
            seq = seq_tensor.squeeze(0).T
            reshaped_data.extend(seq)
        reshaped_data_df = pd.DataFrame(reshaped_data)

        umap_embedding = self.umap.transform(reshaped_data_df)

        predicted_numbers = self.rfc.predict(umap_embedding)
        predicted_aa = np.array([def_dict[i] for i in predicted_numbers])

        predicted_seqs_comp = np.reshape(predicted_aa,
                                         (int(len(predicted_aa)/self._max_len),
                                          self._max_len))

        predicted_seqs = np.apply_along_axis(lambda x: ''.join(str(a) for a in x),
                                             1,
                                             predicted_seqs_comp)
        predicted_seqs_list = predicted_seqs.tolist()

        reconstructed_seqs = self._gap_removal(predicted_seqs_list)
        reconstructed_seqs_df = pd.DataFrame({self.seq_type: reconstructed_seqs})

        return reconstructed_seqs_df

    def model_train(self,
                    train_data: pd.DataFrame,
                    device: torch.device,
                    criterion,
                    input_train_seqs: pd.Series,
                    test_data=None):

        self._embds_shape_check(train_data.drop(columns=self.seq_type))

        if test_data is None:
            seq_train_dataloader = self._make_seq_dataloder(inp_seq=train_data[self.seq_type])
            seq_test_dataloader = None

            embds_train_dataloader = self.input_data_process(
                input_data=train_data.drop(columns=self.seq_type)
                )
            embds_test_dataloader = None
        else:
            seq_train_dataloader = self._make_seq_dataloder(inp_seq=train_data[self.seq_type])
            seq_test_dataloader = self._make_seq_dataloder(inp_seq=test_data[self.seq_type])

            embds_train_dataloader = self.input_data_process(
                input_data=train_data.drop(columns=self.seq_type)
                )
            embds_test_dataloader = self.input_data_process(
                input_data=test_data.drop(columns=self.seq_type)
                )

        part_model_train(self,
                         model_type='decoder',
                         seq_train_dataloader=seq_train_dataloader,
                         embds_train_dataloader=embds_train_dataloader,
                         device=device,
                         criterion=criterion,
                         config=self.config,
                         seq_test_dataloader=seq_test_dataloader,
                         embds_test_dataloader=embds_test_dataloader)

        model_output = model_process(self,
                                     inp_dataloader=embds_train_dataloader,
                                     device=device)

        print(f"Start UMAP training for {self.seq_type}")

        labels = {x: i for i, x in enumerate(KIDERA_DICT.keys())}

        input_train_seqs = input_train_seqs.apply(self.encoder._gap_insertion)
        inp_seqs = input_train_seqs.str.cat(sep='')

        model_out_reshaped_labeled = []
        for num, seq_tensor in enumerate(model_output):
            seq = seq_tensor.squeeze(0).T
            model_out_reshaped_labeled.extend(
                  np.column_stack([seq,
                                   np.array(
                                       [labels[inp_seqs[num*self._max_len + x]] for x in range(self._max_len)]
                                       )]))

        df = pd.DataFrame(model_out_reshaped_labeled)
        df[10] = df[10].astype(int)
        target = df[10]

        df_sampled = df.groupby(10, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), 20000), random_state=42)
        )

        self.umap = UMAP(n_neighbors=10, min_dist=0.3, n_jobs=-1, verbose=False).fit(df_sampled)
        umap_embedding = self.umap.transform(df)
        print(f"UMAP training for {self.seq_type} finished!")

        print(f"Start RandomForestClassifier training for {self.seq_type}")
        X_train, X_test, y_train, y_test = train_test_split(
            umap_embedding, target, test_size=0.3, random_state=42
        )

        self.rfc = RandomForestClassifier(n_estimators=20)
        self.rfc.fit(X_train, y_train)
        print(f"RandomForestClassifier training for {self.seq_type} finished!")

        print("Classification report for full model:\n")
        print(classification_report(y_test, self.rfc.predict(X_test)))

    def save_model(self, output_dir):

        saving_weights(self, output_dir, self.embd_type, self.seq_type)

        umap_suffix = f'umap_{self.embd_type}_{self.seq_type}.pkl'
        umap_path = output_dir.joinpath(umap_suffix)
        with open(umap_path, "wb") as f:
            pickle.dump(self.umap, f)

        rfc_suffix = f'rfc_{self.embd_type}_{self.seq_type}.pkl'
        rfc_path = output_dir.joinpath(rfc_suffix)
        with open(rfc_path, "wb") as f:
            pickle.dump(self.rfc, f)

    def _gap_removal(self, seq_output_list):

        seq_output_list_no_gap = []
        for seq in seq_output_list:
            seq_output_list_no_gap.append(seq.replace('-', ''))

        return seq_output_list_no_gap

    def _embds_shape_check(self, data):
        if (data.shape[1]) != self.latent_dims:
            raise ValueError('Wrong embeddings shape')

    def _make_seq_dataloder(self, inp_seq: pd.Series):

        data = inp_seq.copy()

        data = data.apply(self._gap_insertion)
        data_tensor = torch.tensor(
            np.stack(data
                     .map(lambda seq: self._sequence_to_factor(seq))
                     .values,
                     axis=0,
                     ),
            dtype=torch.float32,
        ).unsqueeze(1)

        seq_dataset = TensorDataset(data_tensor)

        seq_dataloader = DataLoader(seq_dataset,
                                    batch_size=self.config['BATCH_SIZE'],
                                    shuffle=False)

        return seq_dataloader

    def _gap_insertion(self, inp_seq: str) -> str:

        """
        Insert gaps (-) to make all sequences with max length
        Args:
            inp_seq: Amino acid sequence
        """

        if self.seq_type == 'antigen_epitope':
            start_end = (self._max_len - len(inp_seq)) // 2
            if len(inp_seq) % 2 == 0:
                return start_end * "-" + inp_seq + start_end * "-"
            else:
                return (
                    start_end * "-"
                    + inp_seq[: len(inp_seq) // 2]
                    + "-"
                    + inp_seq[len(inp_seq) // 2:]
                    + start_end * "-"
                       )

        length_x = self._max_len - len(inp_seq)
        if len(inp_seq) == 4:
            return inp_seq[:2] + "-" * 15 + inp_seq[2:]
        elif len(inp_seq) == 5:
            return inp_seq[:2] + "-" * 7 + inp_seq[2] + "-" * 7 + inp_seq[3:]
        elif len(inp_seq) == 6:
            return inp_seq[:3] + "-" * 13 + inp_seq[3:]
        else:
            pref, suff = inp_seq[:3], inp_seq[-3:]
            mid = inp_seq[3:-3]
            return (
                pref
                + "-" * (length_x // 2 + length_x % 2)
                + mid
                + "-" * (length_x // 2)
                + suff
            )

    def _sequence_to_factor(self, sequence: str) -> np.ndarray:
        """Convert amino acid sequence to Kidera factors."""
        return np.array([KIDERA_DICT[aa] for aa in sequence],
                        dtype=np.float32).T
