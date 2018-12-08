#default parameters for now, will change them later
import torch

#learning parameters
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001

#constants
MNIST_X = 28
MNIST_Y = 28
MNIST_IM_SIZE = MNIST_X * MNIST_Y

# autoencoders parameters
LATENT = 20

random_seed = 42
torch.manual_seed(random_seed)

