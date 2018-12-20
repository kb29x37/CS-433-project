#### CS433 Project 2 - ICLR Reproducibility Challenge

This project is our submission to Option C of the second project of Machine Learning (CS433), the ICLR Reproducibility Challenge. We chose to reproduce the paper [*MAE: Mutual Posterior-Divergence Regularization for Variational AutoEncoders*](https://openreview.net/forum?id=Hke4l2AcKQ) (our github issue can be found [here](https://github.com/reproducibility-challenge/iclr_2019/issues/93)).

We have decided to use [pytorch](https://pytorch.org) for this project. Our code is structured into the following files :

 - `models.py` contains most of the interesting code. We have implemented the following models :
   - `VAE` is the standard VAE model as described in the paper from [*Kingma & Welling*](https://arxiv.org/abs/1312.6114)
   - `VAE_conv_mnist` follows the same principle but uses ConvNets as encoder and decoder, and expects inputs of the same dimensions as the `MNIST` dataset.
   - `MAE` 
 - `load_data.py` contains helper functions that will load the `MNIST` and `CIFAR10` datasets through pytorch. It will download those data into the `./data` folder if not already present.
 - `hyperparameters.py` contains all hyperparameters for all models
 - `main.py` 

