import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import *

# basic VAE with encoder / decoder as feedforward nets
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(MNIST_IM_SIZE, int(MNIST_IM_SIZE / 2))
        self.fc11 = nn.Linear(MNIST_IM_SIZE, int(MNIST_IM_SIZE / 4))
        self.fc2 = nn.Linear(int(MNIST_IM_SIZE / 2), int(MNIST_IM_SIZE / 4))
        self.fc31 = nn.Linear(int(MNIST_IM_SIZE / 4), LATENT)
        self.fc32 = nn.Linear(int(MNIST_IM_SIZE / 4), LATENT)
        self.fc4 = nn.Linear(LATENT, int(MNIST_IM_SIZE / 4))
        self.fc44 = nn.Linear(LATENT, int(MNIST_IM_SIZE / 2))
        self.fc5 = nn.Linear(int(MNIST_IM_SIZE / 4), int(MNIST_IM_SIZE / 2))
        self.fc6 = nn.Linear(int(MNIST_IM_SIZE / 2), MNIST_IM_SIZE)

    def encode(self, input):
        enc = F.relu(self.fc11(input))
        #enc = F.relu(self.fc2(enc))
        mu = F.relu(self.fc31(enc))
        logvar = F.relu(self.fc32(enc)) # logvar
        return mu, logvar

    def decode(self, latent):
        dec = F.relu(self.fc44(latent))
        #dec = F.relu(self.fc5(dec))
        dec = F.relu(self.fc6(dec))
        return torch.sigmoid(dec) # get values between 0,1 ? (needed for loss but transform?)

    def forward(self, input):
        mu, logvar = self.encode(input)
        noisy_latent = torch.randn_like(mu).mul(torch.exp(0.5*logvar)).add(mu) # pick random numbers
        image_res = self.decode(noisy_latent)
        return image_res, mu, logvar

    # directly from VAE paper, appendix B
    def ELBO_loss(self, mu, logvar, x, y):
        rec_loss = F.binary_cross_entropy(y, x.view(BATCH_SIZE, -1), reduction='sum')
        D_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return rec_loss + D_KL

