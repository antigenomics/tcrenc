# TCR and epitopes autoencoders
This tool containes pretrained autoencoders for complementary determining region 3 (CDR3) domain of the beta chain of human T-cell receptor (TCR) and for epitopes. Sequnce could be representated by one-hot encoding or by Kidera factors.

## Installation

## Usage

## Final models

Final models for one-hot representation could be find in `modules/modules_one-hot/autoencoder.py`.
Weights could be found [here](https://github.com/antigenomics/tcrenc/tree/main/models/models_onehot)


## Results
The results are divided into two folders based on representation. See the `results` folder.

### One hot representation
For the one-hot representation, a comparison of training efficiency was made based on reports generated in the relevant Jupyter notebook (see the `code` folder).

For CDR3, it was shown that single linear transformations (SLT) of the one-hot matrix with a size of (21, 19) (reshaped into a one-dimensional vector) worked well with a latent space size of 64. We also observed that reducing the latent space to a lower dimension, for example, 32, required more complex neural network (NN) architectures, consisting of sequential linear transformations and ReLU activation functions.Two loss functions (MSE and cross-entropy) used in the autoencoder training process for CDR3 sequences were compared. It was demonstrated that the cross-entropy loss function performed approximately 2.5 times better in highly variable positions within CDR3 sequences.

#### VDJdb reconstruction
Autoencoders showed very good performance in generating embeddings and reconstructing sequences on VDJdb.
![reconstruction](https://github.com/antigenomics/tcrenc/blob/main/assets/val_onehot.png)

#### Affinity predictor
The embeddings were used to train affinity predictors. The best model based on robust scaling, PCA transformation, and SVC achieved an ROC-AUC score of **0.65.**
![roc](https://github.com/antigenomics/tcrenc/blob/main/assets/roc_onehot.png)


## Requirements and testings
All requirements could be found in special folder `requirements`. 
There are 2 different dependencies list: 
- for One-Hot representation
- for Kidera factors representation

All scripts were tested on aldan3.itm-rsmu server. One-hot was also tested on MacBook Pro (M1 Pro).

*Note: Pytorch now available only via `pip`. See [this](https://pytorch.org) for more details.*

## References
Goncharov, M., Bagaev, D., Shcherbinin, D., Zvyagin, I., Bolotin, D., Thomas, P. G., Minervina, A. A., Pogorelyy, M. V., Ladell, K., McLaren, J. E., Price, D. A., Nguyen, T. H., Rowntree, L. C., Clemens, E. B., Kedzierska, K., Dolton, G., Rius, C. R., Sewell, A., Samir, J., … Shugay, M. (2022). VDJdb in the pandemic era: A compendium of T cell receptors specific for SARS-COV-2. Nature Methods, 19(9), 1017–1019. https://doi.org/10.1038/s41592-022-01578-0 
