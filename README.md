#### CS433 Project 2 - ICLR Reproducibility Challenge

This project is our submission to Option C of the second project of Machine Learning (CS433), the ICLR Reproducibility Challenge. We chose to reproduce the paper [*MAE: Mutual Posterior-Divergence Regularization for Variational AutoEncoders*](https://openreview.net/forum?id=Hke4l2AcKQ) (our github issue can be found [here](https://github.com/reproducibility-challenge/iclr_2019/issues/93)).

We have decided to use [pytorch](https://pytorch.org) for this project. Our code is structured into the following files :

 - `models.py` contains most of the interesting code. We have implemented the following models :
   - `VAE` is the standard VAE model as described in the paper from [*Kingma & Welling*](https://arxiv.org/abs/1312.6114), using a fully connected network as input encoder and decoder. The structure is the same for all models, and we have one function for the encoding part, one function for the decoding part, one function computing the loss, and the last function computing a forward pass in the network, returning the loss and the resulting (batch) of image
   - `VAE_conv_mnist` follows the same principle but uses ConvNets as encoder and decoder, and expects inputs of the same dimensions as the `MNIST` dataset.
   - `MAE` Our first try on the MAE implementation, working a bit but messy, using a fully connected network as input encoder and decoder
   - `MAE_cleaned` Our working implementation of MAE, using the correct version of the loss, plus the final settings we used in order to generate the images, using a fully connected network as input encoder and decoder
   - `MAE_conv_mnist` An implementation of MAE using convnet as the encoder / decoder
 - `load_data.py` contains helper functions that will load the `MNIST` and `CIFAR10` datasets through pytorch. It will download those data into the `./data` folder if not already present.
 - `main.py` Contains the main code, foar saving / loading models, training the models, and generating some images
 - `k_means` An implementation of the k_means algorithm. It works on the latent space of the model, so on the z variable. I try to create some clusters on the training set, and then once done, tries to fit the test_set data on those clusters
 - `tester.py` plot the losses, try to detect overfitting, get some pictures out of the model using the test_set data
 - `trainer.py` train the models, using Adam as the optimiser. 
 - `utils.py` utility functions used mostly for plots / getting result samples:
   - show_model_result_enc_dec: feed some images to the model, and show the original and resulting image next to each other
   - show_model_result_z_fully_connected: generate some images on fully connected models directly using the latent variable z
   - show_model_result_z_convenet: same as the previous one, but using the convenet model
   - linear_exploration_latent: start from a given value as the latent variable z, and explores the latent space in one given direction, plot the resulting images
   - bilinear_exploration_latent: same as the previou one, but produce a 2D images following 2 directions from 1 point
   - save model: save a trained model in order to gain a lot of time bu not having to retrain all the time
   - load_model: load some saved models:
 - `hyperparameters.py` contains all hyperparameters for all models
   - Models were trained during 10 epochs
   - BATCH_SIZE of training set is 64
   - TEST_batch_size is BATCH_SIZE * 4
   - LR = 0.001
   - LATENT_SPACE = 15
   - eta = 1
   - gamma = 0.01
   - k means K is 10
   - K means iterations is 20
   

