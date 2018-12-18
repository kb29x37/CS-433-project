#default parameters for now, will change them later
import torch

#learning parameters
EPOCHS = 4
BATCH_SIZE = 64
LR = 0.001

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
eta = 0.001
gamma = 0.001

random_seed = 42
torch.manual_seed(random_seed)

