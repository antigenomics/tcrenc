# Autoencoders for TCR CDR3 and epitope sequences

**TCRenc** - tool for training and using autoencoder models to obtain embeddings of amino-acid sequences of complementarity-determining region 3 **(CDR3)** domain of the beta chain of human T-cell receptor **(TCR)** and antigen **epitopes**. This tool supports training, validation, and inference, and also allows using the encoder and decoder separately. Implemented sequence representations include **one-hot encoding** and **Kidera factors**; the resulting embeddings can be used for downstream analyses (clustering, nearest neighbors, classification, etc.).

## Installation

Install by cloning the repository:
```{bash}
git clone https://github.com/antigenomics/tcrenc.git
cd ./tcrenc
conda create -n tcrenc python=3.11.14 
conda activate tcrenc
pip install .     
```

## Usage

There are three entry-point scripts:
* `tcrenc-run` - generate embeddings or reconstruct sequences from embeddings
* `tcrenc-train` - train a model
* `tcrenc-validate` - validate a model on a dataset

All scripts support running a full autoencoder as well as using the encoder or decoder separately. The tool can be used either as a CLI or as an importable Python library. Model hyperparameters, data filtering parameters, and other options are defined in the [configuration files](). New models can be added by following the instructions [here]().


### Script Arguments Documentation

#### `tcrenc-run` - Making embeddings from TCR or epitope sequences or reconstructing sequences from embeddings

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--input` | str | Yes | "VDJdb" option* or path to input CSV file(sequence type should be specified in column name) |
| `--output` | str | Yes | Path to output directory |
| `--embed_type` | str | Yes | Type of sequence representation (see available options below) |
| `--decoder` | flag | No | Use decoder part only (default: false) |
| `--cdr` | flag | No | Specify decoder sequence type as CDR3 (default: false) (Required if you use `--decoder` option) |
| `--epitope` | flag | No | Specify decoder sequence type as epitope (default: false) (Required if you use `--decoder` option)|

\* - "VDJdb" option fetches both CDR3 and antigen_epitope from database, filter it simultaneously and separately make two final CSV files with sequences.

##### Input file format
Input CSV file should contain "cdr3" or "antigen_epitope" columns.
If both sequences types are present in input data they will be processed separately.
All extra data will be ignored.

If `--decoder` option is used input CSV must contain only latent embeddings:
* One sequence = one row
* Number of columns = latent dimensionality (i.e., the embedding size used by the model)
* All values must be numeric (float-compatible)
* Missing values (NA/NaN) are not allowed


##### Output file format
The output is a CSV file with embeddings:
* One sequence = one row
* Number of columns = latent dimensionality (i.e., the embedding size used by the model)
* Last column with original sequence.

If `--decoder` option is used, CSV file with sequences will be produced.


#### `tcrenc-train` - Train models (autoencoder, decoder, encoder) for TCR or epitope sequences

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--input` | str | Yes | "VDJdb" option or path to input CSV file(sequence type should be specified in column name) |
| `--output` | str | Yes | Path to output directory for weights saving |
| `--embed_type` | str | Yes | Type of sequence representation (see available options below) |
| `--weights_save` | flag | No | Save weights (default: false) |
| `--split` | float | No | Split ratio for train/test sets (default: 1 - no split) |
| `--encoder_train` | flag | No | Train only encoder (default: false) |
| `--decoder_train` | flag | No | Train only decoder (default: false) |
| `--cdr` | flag | No | Specify VDJdb sequence type as CDR3 |
| `--epitope` | flag | No | Specify VDJdb sequence type as epitope |


##### Input file format

Input CSV file should contain "cdr3" or "antigen_epitope" columns.
If both sequences types are presented in the input data they will be processed separately.
All extra data will be ignored.

##### Output file format

As output there will be files, which produced by model `save_model()` method. (Weights for PyTorch models (`.pth`) in presented models)

#### `tcrenc-validate` - Validate model on input TCR or epitope sequences or on VDJdb

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--input` | str | Yes | "VDJdb" option or path to input CSV file(sequence type should be specified in column name) |
| `--output` | str | Yes | Path to output directory |
| `--embed_type` | str | Yes | Type of sequence representation (see available options below) |
| `--decoder` | flag | No | Use decoder only (default: false) |
| `--cdr` | flag | No | Specify VDJdb sequence type as CDR3 |
| `--epitope` | flag | No | Specify VDJdb sequence type as epitope |

##### Input file format
Input CSV file should contain "cdr3" or "antigen_epitope" columns.
If both sequences types are presented in the input data they will be processed separately.
All extra data will be ignored.

##### Output file format
As output there will be:
* PDF report with error metrics
* Main report in YAML format with error metrics
* CSV files with wrong sequences (wrong sequences with right length, wrong sequences with wrong length)



### Available Embedding Types (`--embed_type`)

The following embedding types are available for all scripts:
- `onehot` - One-hot encoding representation
- `kidera` - Kidera factors representation

### Sequence Type Specification

Note about sequence type flags (`--cdr` and `--epitope`):
- These are mutually exclusive flags
- One must be specified when working with VDJdb data (in `tcrenc-train` or `tcrenc-validate`)
- For custom CSV files, the sequence type is determined by the column headers

### Examples
Train autoencoder model using kidera factors sequence representation:
```{bash}
tcrenc-train --input VDJdb --output ./testing --embed_type kidera --weights_save
```

Train decoder only based on data presented in CSV file `./testing/embeddings_cdr3_onehot.csv` with train/test split:
```
tcrenc-train --input ./testing/embeddings_cdr3_onehot.csv --output ./testing --embed_type onehot --decoder_train --split 0.8
```

Validate pretrained model on VDJdb (weights should be specified in configuration file):
```
tcrenc-validate --input VDJdb --output ./testing --embed_type onehot  
```

Make embeddings:
```{bash}
tcrenc-run --input VDJdb --embed_type onehot --output ./testing
```

**Other Usage examples could be found [here]().**

## Results

This tool was used to train autoencoder models using **one-hot** and **Kidera factors** representations of the input sequences. The model architectures are described in the [Models section](). Pretrained weights for the **one-hot** autoencoder are also provided in this repository.

### One-hot model
- CDR3 sequence reconstruction accuracy on VDJdb: **99.3%**
- Antigen epitope sequence reconstruction accuracy on VDJdb: **99.9%**
- Best binding predictor ROC AUC: **0.6456**

### Kidera factors model
- CDR3 sequence reconstruction accuracy on VDJdb: **47.5%**
- Antigen epitope sequence reconstruction accuracy on VDJdb: **10.6%**
- Best binding predictor ROC AUC: **0.6282**


## References
Goncharov, M., Bagaev, D., Shcherbinin, D., Zvyagin, I., Bolotin, D., Thomas, P. G., Minervina, A. A., Pogorelyy, M. V., Ladell, K., McLaren, J. E., Price, D. A., Nguyen, T. H., Rowntree, L. C., Clemens, E. B., Kedzierska, K., Dolton, G., Rius, C. R., Sewell, A., Samir, J., … Shugay, M. (2022). VDJdb in the pandemic era: A compendium of T cell receptors specific for SARS-COV-2. Nature Methods, 19(9), 1017–1019. https://doi.org/10.1038/s41592-022-01578-0 