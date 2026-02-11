# Adding New Models to TCRenc

## Introduction

**TCRenc** is organized around three CLI entry points — `tcrenc-train`, `tcrenc-validate`, and `tcrenc-run` — covering the full autoencoder workflow: training, evaluation, and inference (either generating embeddings or reconstructing sequences from latent vectors).

The core design principle is that **all model-specific logic is implemented inside the model itself**: loading weights, encoding sequences into embeddings (encoder), decoding embeddings back into sequences (decoder), as well as training and validation routines. The scripts act as a thin wrapper: they parse user options, load the appropriate model, and call the corresponding model methods.

All customizable settings (model hyperparameters, paths to weights, data filtering options, device/GPU settings, and other runtime parameters) are moved into **YAML configuration files**.

## General considerations

### Configuration
In **TCRenc**, model behavior is controlled via [configuration files](), which are standard **YAML** files. There are two config types:

- **general**: global TCRenc settings (device/GPU, sequence filtering parameters, etc.)
- **model**: model-specific settings (training/validation/run parameters)

A model config file may include the following sections:
- `general`
- `train` (required if the model will be trained)
- `validate` (required if the model will be validated)
- `run` (required if the model will be used to generate embeddings or decode them)

### Indexing
To index a new model, add its name to the `AV_EMBD_TYPE` list in [this Python file](). Then update:
- `load_model_for_train()`
- `load_model_for_run()`
- `load_model_for_validate()`
so they can instantiate and load your model.


## Required model interface

Model should contain `__init__(self, config: dict, seq_type: str, device: torch.device)` method. 

### `tcrenc-run`
Your model should implement:
- `weight_load()` — loads model weights (may be a no-op if not applicable)
- `make_embeddings_from_seq(input_data: pd.DataFrame) -> pd.DataFrame` — encoder: converts sequences to embeddings
- `make_seq_from_embeddings(input_embds: pd.DataFrame) -> pd.DataFrame` — decoder: reconstructs sequences from embeddings

### `tcrenc-train`
Your model should implement:
- `model_train(train_data: pd.DataFrame, criterion, test_data: pd.DataFrame | None = None) -> None` — trains the model
- `save_model(output_path: Path) -> None` — saves model weights

### `tcrenc-validate`
Your model should implement:
- `weight_load()` — loads model weights (may be a no-op)
- `validation_on_seqs(input_data: pd.DataFrame, loss_function) -> (input_seqs, output_seqs, loss_value)` — runs validation (full model or decoder-only) on the provided sequences
