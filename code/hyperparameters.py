#default parameters for now, will change them later
import torch

#learning parameters
EPOCHS = 10
BATCH_SIZE = 64
TEST_BATCH_SIZE = BATCH_SIZE * 4
LR = 0.001
LR_K_MEANS = 0.1

#constants
MNIST_X = 28
MNIST_Y = 28
MNIST_IM_SIZE = MNIST_X * MNIST_Y

# autoencoders parameters
LATENT = 15

# convolutional parameters
IN_CHANNELS_CIFAR = 3
IN_CHANNELS_MNIST = 1
COMPL = 10

# MAE
eta = 1.0
gamma = 0.01

# N_SAMPLES latent
LATENT_SAMPLES = 10

# K_means parameteres
K = 10
K_MEANS_EPOCHS = 20
threshold = 1e-5


random_seed = 42
torch.manual_seed(random_seed)

