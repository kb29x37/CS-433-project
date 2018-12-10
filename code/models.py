import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as d

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
        self.fc61 = nn.Linear(int(MNIST_IM_SIZE / 2), MNIST_IM_SIZE)
        self.fc62 = nn.Linear(int(MNIST_IM_SIZE / 2), MNIST_IM_SIZE)


        self.rec_loss = nn.MSELoss()
        #self.rec_loss = nn.BCELoss() #-> good if binary data
        #self.rec_loss = nn.NLLLoss()

    def encode(self, input):
        enc = F.relu(self.fc11(input))
        #enc = F.relu(self.fc2(enc))
        mu = F.relu(self.fc31(enc))
        sigma = F.relu(self.fc32(enc)) # logvar might be needed if overflow
        return mu, sigma

    def decode(self, latent):
        dec = F.relu(self.fc44(latent))
        #dec = F.relu(self.fc5(dec))
        dec = F.relu(self.fc61(dec))
        #mu = F.relu(self.fc61(dec))
        #sigma = F.relu(self.fc62(dec))
        #dec = torch.randn_like(mu).mul(sigma).add(mu) # eventually second reparametrisation
        return torch.sigmoid(dec)

    def forward(self, input):
        mu, sigma = self.encode(input)
        noisy_latent = torch.randn_like(mu).mul(sigma).add(mu) # reparametrisation trick from distribution
        image_res = self.decode(noisy_latent)
        return image_res, mu, sigma

    # directly from VAE paper, appendix B
    def ELBO_loss(self, mu, sigma, x, y):
        rec_loss = self.rec_loss(y, x.view(BATCH_SIZE, -1)) # why this one work?
        #rec_loss = self.rec_loss(y, x.view(BATCH_SIZE, -1).type(torch.LongTensor)) # why this one work?
        #normal_dists = torch.zeros(BATCH_SIZE)
        #for i in range(0, BATCH_SIZE):
        #    n_dist =  d.multivariate_normal.MultivariateNormal(mu_d[i], torch.diag(sigma_d[i]))
        #    print(y[i])
        #    normal_dists[i] = n_dist.log_prob(y[i])
        #rec_loss = torch.mean(normal_dists)
        D_KL = 0.5 * torch.sum(-1 -torch.log(1e-8 + sigma.pow(2)) + mu.pow(2) + sigma.pow(2))
        return (rec_loss + D_KL)

class VAE_mu_var(nn.Module):
    def __init__(self):
        super(VAE_mu_var, self).__init__()
        self.enc_fc1 = nn.Linear(MNIST_IM_SIZE, int(MNIST_IM_SIZE/2))
        self.enc_fcmu = nn.Linear(int(MNIST_IM_SIZE/2), LATENT)
        self.enc_fclogvar = nn.Linear(int(MNIST_IM_SIZE/2), LATENT)

        self.dec_fc1 = nn.Linear(LATENT, int(MNIST_IM_SIZE/2))
        self.dec_fcmu = nn.Linear(int(MNIST_IM_SIZE/2), MNIST_IM_SIZE)
        self.dec_fclogvar = nn.Linear(int(MNIST_IM_SIZE/2), MNIST_IM_SIZE)

    def nll_loss(self, y_gen, mu, logvar):
        res = torch.zeros(mu.size(0))
        for i in range(0, mu.size(0)):
            dist = d.multivariate_normal.MultivariateNormal(mu[i], torch.diag(torch.exp(0.5*logvar[i])))
            res[i] = dist.log_prob(y_gen[i])
        return -torch.mean(res)

    def KL_reguarisation(self, mu_z, logvar_z):
        return 0.5 * torch.sum(mu.pow(2) + torch.exp(logvar) - torch.log(1e-8 + sigma.pow(2)) - 1)

    def encode(self, input):
        h = F.relu(self.enc_fc1(input))
        mu = F.relu(self.enc_fcmu(h))
        logvar = F.relu(self.enc_fclogvar(h))
        return mu, logvar

    def decode(self, z):
        h = F.relu(self.dec_fc1)
        mu = F.relu(self.dec_fcmu(h))
        logvar = F.relu(self.dec_fclogvar(h))
        image_res = torch.sigmoid(torch.randn_like(mu).mul(torch.exp(0.5 * logvar)).add(mu))
        return image_res, logvar, mu

    def forward(self, input):
        mu_z, logvar_z = self.encode(input)
        noisy_z = torch.randn_like(mu).mul(torch.exp(0.5 * logvar)).add(mu_z) # reparametrisation
        image_res, mu_y, logvar_y = self.decode(noisy_z)
        loss = nll_loss(mu_y, logvar_y) + KL_regularisation(mu_z, logvar_z)
        return image_res, loss

class VAE_conv_mnist(nn.Module):
    def __init__(self):
        super(VAE_conv, self).__init__()
        self.loss = nn.BCELoss() # BCE loss suggested from pytorch example for convnets, check why

        # conv(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv1 = nn.Conv2d(IN_CHANNELS_MNIST, 1 * COMPL, 5, 1, 1) # not tested, need to check all dimensions
        self.conv2 = nn.Conv2d(1 * COMPL, 2 * COMPL, 5, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(2 * COMPL)
        self.conv3 = nn.Conv2d(2 * COMPL, 4 * COMPL, 3, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(4 * COMPL)
        self.conv4 = nn.Conv2d(4 * COMPL, 8 * COMPL, 3, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(8 * COMPL)
        self.conv5_mu = nn.Conv2d(8 * COMPL, 1, 3, 1, 0)
        self.conv5_logvar = nn.Conv2d(8 * COMPL, 1, 3, 1, 0)

        # TODO eventually add fully connected layers at the end

        self.deconv1 = nn.ConvTranspose2d(LATENT, 8 * COMPL, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(8 * COMPL)
        self.deconv2 = nn.ConvTranspose2d(8 * COMPL, 4 * COMPL, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(4 * COMPL)
        self.deconv3 = nn.ConvTranspose2d(4 * COMPL, 2 * COMPL, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(2 * COMPL)
        self.deconv4 = nn.ConvTranspose2d(2 * COMPL, 1 * COMPL, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(1 * COMPL)
        self.deconv5 = nn.ConvTranspose2d(1 * COMPL, 1, 4, 2, 1)

        self.weights_init(mean, std)# seems to avoid vanishing / exploding gradients

    def encode(self, input):
        x_temp = self.conv1(input)
        x_temp = self.conv2(x_temp)
        x_temp = self.conv2_bn(x_temp)
        x_temp = self.conv3(x_temp)
        x_temp = self.conv3_bn(x_temp)
        x_temp = self.conv4(x_temp)
        x_temp = self.conv4_bn(x_temp)
        mu = self.conv5_mu(x_temp)
        logvar = self.conv5_logvar(x_temp)
        return mu, logvar

    def decode(self, z):
        z_temp = self.deconv1(z)
        z_temp = self.deconv1_bn(z_temp)
        z_temp = self.deconv2(z_temp)
        z_temp = self.deconv2_bn(z_temp)
        z_temp = self.deconv3(z_temp)
        z_temp = self.deconv3_bn(z_temp)
        z_temp = self.deconv4(z_temp)
        z_temp = self.deconv4_bn(z_temp)
        z_temp = self.deconv5(z_temp)
        return torch.sigmoid(z_temp) # needed?

    def loss(self, mu, logvar, x, y_gen):
        rec_loss = self.loss(x, y_gen)
        D_KL = 0.5 * torch.sum(-1 -torch.log(1e-8 + sigma.pow(2)) + mu.pow(2) + sigma.pow(2))
        return rec_loss + D_KL

    def forward(self, input):
        mu_z, logvar_z = self.encode(input)
        noisy_latent = torch.randn_like(mu_z).mul(torch.exp(0.5*logvar)).add(mu) # reparametrisation
        image = self.decode(noizy_latent)
        return loss(mu_z, logvar_z, input, y_gen), image

