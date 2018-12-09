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

        self.rec_loss = nn.MSELoss()
        #self.rec_loss = nn.BCELoss()

    def encode(self, input):
        enc = F.relu(self.fc11(input))
        #enc = F.relu(self.fc2(enc))
        mu = F.relu(self.fc31(enc))
        sigma = F.relu(self.fc32(enc)) # logvar might be needed if overflow
        return mu, sigma

    def decode(self, latent):
        dec = F.relu(self.fc44(latent))
        #dec = F.relu(self.fc5(dec))
        dec = F.relu(self.fc6(dec))
        return torch.sigmoid(dec) # get values between 0,1 ? (needed for loss but transform?)

    def forward(self, input):
        mu, sigma = self.encode(input)
        noisy_latent = torch.randn_like(mu).mul(sigma).add(mu) # reparametrisation trick from distribution
        image_res = self.decode(noisy_latent)
        return image_res, mu, sigma

    # directly from VAE paper, appendix B
    def ELBO_loss(self, mu, sigma, x, y):
        rec_loss = self.rec_loss(y, x.view(BATCH_SIZE, -1)) # why this one work?
        D_KL = 0.5 * torch.sum(-1 -torch.log(1e-8 + sigma.pow(2)) + mu.pow(2) + sigma.pow(2))
        return (rec_loss + D_KL)

