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
        self.fc11 = nn.Linear(MNIST_IM_SIZE, int(MNIST_IM_SIZE * 4))
        self.fc2 = nn.Linear(int(MNIST_IM_SIZE / 2), int(MNIST_IM_SIZE / 4))
        self.fc31 = nn.Linear(int(MNIST_IM_SIZE * 4), LATENT)
        self.fc32 = nn.Linear(int(MNIST_IM_SIZE * 4), LATENT)
        self.fc4 = nn.Linear(LATENT, int(MNIST_IM_SIZE / 4))
        self.fc44 = nn.Linear(LATENT, int(MNIST_IM_SIZE / 2))
        self.fc5 = nn.Linear(int(MNIST_IM_SIZE / 4), int(MNIST_IM_SIZE / 2))
        self.fc61 = nn.Linear(int(MNIST_IM_SIZE / 2), MNIST_IM_SIZE)
        self.fc62 = nn.Linear(int(MNIST_IM_SIZE / 2), MNIST_IM_SIZE)


        #self.rec_loss = nn.MSELoss(reduction='sum')
        self.rec_loss = nn.BCELoss(reduction='sum') # the sum here is super important
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
        return image_res, self.ELBO_loss(mu, sigma, input, image_res)

    # directly from VAE paper, appendix B
    def ELBO_loss(self, mu, sigma, x, y):
        rec_loss_res = self.rec_loss(y, x.view(BATCH_SIZE, -1))
        #rec_loss = self.rec_loss(y, x.view(BATCH_SIZE, -1).type(torch.LongTensor)) # why this one work?
        #normal_dists = torch.zeros(BATCH_SIZE)
        #for i in range(0, BATCH_SIZE):
        #    n_dist =  d.multivariate_normal.MultivariateNormal(mu_d[i], torch.diag(sigma_d[i]))
        #    print(y[i])
        #    normal_dists[i] = n_dist.log_prob(y[i])
        #rec_loss = torch.mean(normal_dists)

        #should be the sum over dimesion of z
        D_KL = torch.sum(0.5 * torch.sum(-1 -torch.log(1e-8 + sigma.pow(2)) + mu.pow(2) + sigma.pow(2), dim=1))
        #print(D_KL.size())
        return (rec_loss_res + D_KL)

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
        return torch.sum(0.5 * torch.sum(mu.pow(2)
                                         + torch.exp(logvar) - torch.log(1e-8 + torch.exp(logvar)) - 1, dim=1))

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

class VAE_conv_mnist(nn.Module): # TODO find suitable hyper parameters
    def __init__(self):
        super(VAE_conv_mnist, self).__init__()
        self.loss = nn.BCELoss(reduction='sum') # BCE loss suggested from pytorch example for convnets, check why

        # conv(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv1 = nn.Conv2d(IN_CHANNELS_MNIST, 1 * COMPL, 5, 1, 1) # not tested, need to check all dimensions
        self.conv2 = nn.Conv2d(1 * COMPL, 2 * COMPL, 5, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(2 * COMPL)
        self.conv3 = nn.Conv2d(2 * COMPL, 4 * COMPL, 3, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(4 * COMPL)
        self.conv4 = nn.Conv2d(4 * COMPL, 8 * COMPL, 3, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(8 * COMPL)
        self.conv5_mu = nn.Conv2d(8 * COMPL, LATENT, 3, 1, 0)
        self.conv5_logvar = nn.Conv2d(8 * COMPL, LATENT, 3, 1, 0)
        self.enc_out_mu = nn.Linear(int(8 * COMPL), LATENT)
        self.enc_out_logvar = nn.Linear(int(8 * COMPL), LATENT)

        # TODO eventually add fully connected layers at the end

        self.deconv1 = nn.ConvTranspose2d(LATENT, 8 * COMPL, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(8 * COMPL)
        self.deconv2 = nn.ConvTranspose2d(8 * COMPL, 4 * COMPL, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(4 * COMPL)
        self.deconv3 = nn.ConvTranspose2d(4 * COMPL, 2 * COMPL, 2, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(2 * COMPL)
        self.deconv4 = nn.ConvTranspose2d(2 * COMPL, 1 * COMPL, 2, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(1 * COMPL)
        self.deconv5 = nn.ConvTranspose2d(1 * COMPL, 1, 5, 1, 1)
        self.dec_out = nn.Linear(int(80), MNIST_IM_SIZE)


    def encode(self, input):
        x_temp = F.relu(self.conv1(input))
        #print(x_temp.size())
        x_temp = F.relu(self.conv2(x_temp))
        #print(x_temp.size())
        x_temp = self.conv2_bn(x_temp)
        x_temp = F.relu(self.conv3(x_temp))
        #print(x_temp.size())
        x_temp = self.conv3_bn(x_temp)
        x_temp = F.relu(self.conv4(x_temp))
        #print(x_temp.size())
        x_temp = self.conv4_bn(x_temp)
        mu = F.relu(self.conv5_mu(x_temp))
        #print(mu.size())
        logvar = F.relu(self.conv5_logvar(x_temp))
        #print(logvar.size())
        #mu = F.relu(self.enc_out_mu(mu))
        #print(mu.size())
        #logvar = F.relu(self.enc_out_logvar(logvar))
        #print(logvar.size())
        return mu, logvar

    def decode(self, z):
        print(z.size())
        z_temp = F.relu(self.deconv1(z))
        #print(z_temp.size())
        z_temp = self.deconv1_bn(z_temp)
        z_temp = F.relu(self.deconv2(z_temp))
        #print(z_temp.size())
        z_temp = self.deconv2_bn(z_temp)
        z_temp = F.relu(self.deconv3(z_temp))
        #print(z_temp.size())
        z_temp = self.deconv3_bn(z_temp)
        z_temp = F.relu(self.deconv4(z_temp))
        #print(z_temp.size())
        z_temp = self.deconv4_bn(z_temp)
        z_temp = F.relu(self.deconv5(z_temp))
        #print(z_temp.size())
        return torch.sigmoid(z_temp) # needed?

    def whole_loss(self, mu, logvar, x, y_gen):
        rec_loss = self.loss(y_gen.reshape(BATCH_SIZE, -1), x.reshape(BATCH_SIZE, -1))
        D_KL = torch.sum(0.5 * torch.sum(-1 -torch.log(1e-8 + torch.exp(logvar))
                                         + mu.pow(2) + torch.exp(logvar), dim=1))
        return rec_loss + D_KL

    def forward(self, input):
        mu_z, logvar_z = self.encode(input)
        noisy_latent = torch.randn_like(mu_z).mul(torch.exp(0.5*logvar_z)).add(mu_z) # reparametrisation
        y_gen = self.decode(noisy_latent)
        return self.whole_loss(mu=mu_z, logvar=logvar_z, x=input, y_gen=y_gen), y_gen

# basic VAE with encoder / decoder as feedforward nets
class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
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

        #self.rec_loss = nn.MSELoss(reduction='sum')
        self.rec_loss = nn.BCELoss(reduction='sum') #-> good if binary data ?
        #self.rec_loss = nn.NLLLoss()

    def encode(self, input):
        enc = F.relu(self.fc11(input))
        mu = F.relu(self.fc31(enc))
        sigma = F.relu(self.fc32(enc)) # logvar might be needed if overflow
        return mu, sigma

    def decode(self, latent):
        dec = F.relu(self.fc44(latent))
        dec = F.relu(self.fc61(dec))
        return torch.sigmoid(dec)

    def forward(self, input):
        # separate the input in two distinct
        x_1 = input[0:int(BATCH_SIZE/2)]
        x_2 = input[int(BATCH_SIZE/2):BATCH_SIZE]

        # create two sets of z
        mu_1, sigma_1 = self.encode(x_1)
        mu_2, sigma_2 = self.encode(x_2)

        noisy_latent_1 = torch.randn_like(mu_1).mul(sigma_1).add(mu_1)
        noisy_latent_2 = torch.randn_like(mu_2).mul(sigma_2).add(mu_2)

        image_res_1 = self.decode(noisy_latent_1)
        image_res_2 = self.decode(noisy_latent_2)

        image_res = torch.cat((image_res_1, image_res_2), 0)

        mae_loss = self.MAE_loss(mu_1, sigma_1, mu_2, sigma_2, input, image_res)

        return image_res, mae_loss

    # From MAE paper, applying the two regularisation
    def MAE_loss_pytorch(mu_1, sigma_1, mu_2, sigma_2, x, y):
        batch_size_2 = int(BATCH_SIZE/2)

    def MAE_loss(self, mu_1, sigma_1, mu_2, sigma_2, x, y):
        batch_size_2 = int(BATCH_SIZE/2)

        sigma_1 = torch.add(sigma_1, 1e-8)
        sigma_2 = torch.add(sigma_2, 1e-8)

        mu_1 = torch.add(mu_1, 1e-8)
        mu_2 = torch.add(mu_2, 1e-8)

        #print(sigma_1.size())
        rec_loss = self.rec_loss(y, x.view(BATCH_SIZE, -1)) # why this one work?

        # for the normal loss, we use the mean betwee the mus, sigmas to compute it
        sigma = torch.add(sigma_1, sigma_2) * 0.5
        mu = torch.add(mu_1, mu_2) * 0.5
        print(mu.size())
        D_KL_p_q = 0.5 * torch.sum(-1 -torch.log(1e-8 + sigma.pow(2)) + mu.pow(2) + sigma.pow(2), dim=1)

        cov_1 = torch.autograd.Variable(torch.zeros(batch_size_2, sigma_1.size(1), sigma_1.size(1)))
        cov_2 = torch.autograd.Variable(torch.zeros(batch_size_2, sigma_2.size(1), sigma_2.size(1)))

        #print(cov_1.size())

        cov_1.as_strided(sigma_1.size(), [cov_1.stride(0), cov_1.size(2) + 1]).copy_(sigma_1)
        cov_2.as_strided(sigma_2.size(), [cov_2.stride(0), cov_2.size(2) + 1]).copy_(sigma_2)

        #print(cov_1.size())
        #print(sigma_1[0])

        #print(sigma_1.size())

        inv_cov_1 = torch.inverse(cov_1)
        inv_cov_2 = torch.inverse(cov_2)

        det_cov_1 = torch.zeros(cov_1.size(0))
        det_cov_2 = torch.zeros(cov_2.size(0))

        for i in range(0, batch_size_2):
            det_cov_1[i] = torch.det(cov_1[i])
            det_cov_2[i] = torch.det(cov_2[i])

        #print("det_cov_1: " + str(det_cov_1))
        #print("det_cov_2: " + str(det_cov_2))

        #print(inv_cov_2.size())
        #print(cov_1.size())

        mult = torch.matmul(inv_cov_2, cov_1)

        trace = torch.zeros(batch_size_2)
        for i in range(0, batch_size_2):
            trace[i] = torch.trace(mult[i])

        #print(mu_2)
        #print(mu_1)
        a = torch.transpose((mu_2 - mu_1).resize(batch_size_2, LATENT, 1), dim0=1, dim1=2)
        #print("a:" + str(a))
        #print(a.size())
        #print(inv_cov_2.size())
        b = torch.matmul(a, inv_cov_2)
        #print(b.size())
        #print("b: " + str(b))
        c = torch.add(mu_2, -1, mu_1)
        #print("c: " + str(c))
        d = torch.matmul(b.resize(batch_size_2, 1, LATENT), c.resize(batch_size_2, LATENT, 1))
        #print("d: " + str(d))
        e = torch.log(1e-8 + det_cov_2 / (det_cov_1 + 1e-8)) - mu_1.size(1) #-> dets are all zeros?
        #e = 1 - mu_1.size(1)
        #print("e: " + str(e))

        # L_diverse
        D_KL_q_q = 0.5 * (trace + d + e)
        D_KL_q_q = D_KL_q_q.resize(batch_size_2, batch_size_2)
        #print(D_KL_q_q)
        L_diverse = torch.sum(torch.log(1 + torch.exp(-D_KL_q_q)), dim=1)# check this mean thing here

        # L smooth
        mean_q_q = torch.sum(D_KL_q_q, dim=1)
        #print(mu.size())
        L_smooth = torch.sqrt(torch.sum(D_KL_q_q - mean_q_q, dim=1).pow(2) / (mu_1.size(1) - 1))

        loss = (rec_loss + torch.sum(D_KL_p_q) + eta * torch.sum(L_diverse) + gamma * torch.sum(L_smooth))
        print("rec: " + str(rec_loss) + " D_KL_p_q: " + str(D_KL_p_q) + " L_diverse: " +
              str(L_diverse) + " L_smooth: " + str(L_smooth))
        print("rec: " + str(rec_loss.size()) + " D_KL_p_q: " + str(D_KL_p_q.size()) + " L_diverse: " +
              str(L_diverse.size()) + " L_smooth: " + str(L_smooth.size()))
        return loss

## Resnet implementation from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = strid

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
