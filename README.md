# TCR and epitopes autoencoders
This tool containes pretrained autoencoders for complementary determining region 3 (CDR3) domain of the beta chain of human T-cell receptor (TCR) and for epitopes. Sequnce could be representated by one-hot encoding or by Kidera factors.

## Installation
Install this project via cloning this repository:
```{bash}
git clone git@github.com:antigenomics/tcrenc.git
```
## Usage

The use of this project is to interact with it through the run.py file. This file must be run via the command line. In total, this file takes 3 mandatory and 1 optional argument:
- `input` - requires string with absolute path to the input `.csv` file.
- `output` - requires path to output directory
- `embed_type` - requires one of two options (`onehot` or `kidera`). Sequence representation type
- `residual_block` - requires `true` or `false` (basically `false`). Using residual block layer in convolutional autoencoder (only makes sense if you use kidera factors).

Examples:
```{bash}
python run.py --input ~/tcrenc/dataset/X_test.csv --output . --embed_type onehot 
```
After running this line you will get embeddings for cdr and epitope sequences by using model for one hot representation.
```{bash}
python run.py --input ~/tcrenc/dataset/X_test.csv --output . --embed_type kidera --residual_block true
```
After running this line you will get embeddings for cdr and epitope sequences by using model for kidera representations with Residual Blocks in architecture.

## Format
### Input
The input file should be a csv file with a clear structure. It should have 2 columns named cdr3 and antigen_epitope. Each observation in these columns is a cdr or epitope sequence, respectively. 

### Output
After running our algorithms you will get embeddings of your sequences. The output format depends on the representation you choose.

If you choose OneHot, you will get 4 files with embeddings for tcr and epitopе respectively. The last index in the output file name indicates the index before or after which the dummy amino acids are placed:

- `..3` - dummy amino acids come after the third amino acid
- `..4` - dummy amino acids come after the fourth amino acid
- `..-3` - dummy amino acids come before the third from the end amino acid
- `..-4` - dummy amino acids come before the fourth from the end amino acid

If you choose Kidera, you will get 2 files with embeddings for tcr and epitopе respectively. It is important to note that if you choose to use the Residual block (`residual_block true`), the file name will have the suffix `_residual` at the end. 
## Final models

Final models for one-hot representation could be find in `modules/modules_one-hot/autoencoder.py`.
Weights could be found [here](https://github.com/antigenomics/tcrenc/tree/main/models/models_onehot)

If you want to use weights for models using kidera factors - you need to have access to the aldan server.

## Results
The results are divided into two folders based on representation. See the `results` folder.

### One hot representation
For the one-hot representation, a comparison of training efficiency was made based on reports generated in the relevant Jupyter notebook (see the `code` folder).

For CDR3, it was shown that single linear transformations (SLT) of the one-hot matrix with a size of (21, 19) (reshaped into a one-dimensional vector) worked well with a latent space size of 64. We also observed that reducing the latent space to a lower dimension, for example, 32, required more complex neural network (NN) architectures, consisting of sequential linear transformations and ReLU activation functions.Two loss functions (MSE and cross-entropy) used in the autoencoder training process for CDR3 sequences were compared. It was demonstrated that the cross-entropy loss function performed approximately 2.5 times better in highly variable positions within CDR3 sequences.

#### VDJdb reconstruction
Autoencoders showed very good performance in generating embeddings and reconstructing sequences on VDJdb.
![reconstruction](https://github.com/antigenomics/tcrenc/blob/main/assets/val_onehot.png)

#### Affinity predictor
The embeddings were used to train affinity predictors. The best model based on robyst scaler, PCA transformation, and SVC achieved an ROC-AUC score of **0.65.**
![roc](https://github.com/antigenomics/tcrenc/blob/main/assets/roc_onehot.png)

### Kidera representation

Kidera factors - Kidera factors are a set of ten numerical values that represent the physical properties of amino acids. These factors are useful for characterizing protein sequences and predicting their structural and functional properties.

Our neural network model is a convolutional autoencoder that reduces the dimension of the original representation using sequential convolution operations. In this case, 3 convolutional layers and 2 linear layers are used. As a result, we get a latency space with a size of 64. Then, similarly, we can decode this latent space back into our representations. We concatenated the latent spaces obtained from the cdr sequence and epitope and used them to train a fully connected neural network. Also, in the course of this study, we used the Residual Block to possibly solve the problem of gradient attenuation and the fact that layers can interfere with each other, which causes the generalizing ability to decrease. However, this model showed a worse result than a conventional autoencoder.

We got quite good results for a convolutional autoencoder without a Residual Block. The results with the Residual Block are slightly worse.

We also checked how well the autoencoder captures the sequence structure and individual amino acids. And here he also showed good results.

## Requirements and testings
All requirements could be found in special folder `requirements`. 
There are 2 different dependencies list: 
- for One-Hot representation
- for Kidera factors representation

All scripts were tested on aldan3.itm-rsmu server. One-hot was also tested on MacBook Pro (M1 Pro).

*Note: Pytorch now available only via `pip`. See [this](https://pytorch.org) for more details.*

## References
Goncharov, M., Bagaev, D., Shcherbinin, D., Zvyagin, I., Bolotin, D., Thomas, P. G., Minervina, A. A., Pogorelyy, M. V., Ladell, K., McLaren, J. E., Price, D. A., Nguyen, T. H., Rowntree, L. C., Clemens, E. B., Kedzierska, K., Dolton, G., Rius, C. R., Sewell, A., Samir, J., … Shugay, M. (2022). VDJdb in the pandemic era: A compendium of T cell receptors specific for SARS-COV-2. Nature Methods, 19(9), 1017–1019. https://doi.org/10.1038/s41592-022-01578-0 
