import torch
import numpy as np
import load_data
import trainer
import models
import matplotlib.pyplot as plt

from hyperparameters import *

MODEL_PATH = './model.pt'
SAMPLES = 10

def show_model_result_enc_dec(model, loader):
    image = load_data.get_dataset_image(loader)

    (x, target) = next(iter(loader))

    res_image, loss = model(x.reshape(BATCH_SIZE, -1))

    res_image = res_image.detach().numpy()
    print(x.size())
    for i in range(0, BATCH_SIZE):
        fig = plt.figure(figsize=(2,1))

        fig.add_subplot(1, 2, 1)
        plt.imshow(res_image[i].reshape(MNIST_X, MNIST_Y))
        fig.add_subplot(1, 2, 2)
        plt.imshow(x[i].reshape(MNIST_X, MNIST_Y))
        plt.show()

def show_model_result_z_fully_connected(model, loader):

    for i in range(0, LATENT):
        #noise = torch.randn(LATENT)
        noise = torch.zeros(LATENT)
        noise[i] = 1

        res_image = model.decode(noise.reshape(1, -1))

        res_image = res_image.detach().numpy()
        plt.imshow(res_image[0].reshape(MNIST_X, MNIST_Y))
        plt.show()

def show_model_result_z_convnet(model, loader):

    for i in range(0, LATENT):
        noise = torch.randn(LATENT)
        #noise = torch.zeros(LATENT)
        #noise[i] = 1

        res_image = model.decode(noise.reshape(1, -1, 1, 1))

        res_image = res_image.detach().numpy()
        plt.imshow(res_image[0].reshape(MNIST_X, MNIST_Y))
        plt.show()

def linear_exploration_latent(model,
                              start=torch.tensor([0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]),
                              direction=torch.tensor([0.,-1.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])):
    #direction = torch.randn(LATENT)
    eps = 0.1

    #initial_noise = torch.randn(LATENT)
    initial_noise = start

    fig = plt.figure(figsize=(1,LATENT_SAMPLES))

    for i in range(0, LATENT_SAMPLES):
        print(initial_noise)
        print(direction)
        res_image = model.decode(initial_noise.reshape(1, -1))

        res_image = res_image.detach().numpy()
        fig.add_subplot(1, LATENT_SAMPLES, i+1)
        plt.imshow(res_image[0].reshape(MNIST_X, MNIST_Y))

        initial_noise += direction * eps

    plt.show()

def bilinear_exploration_latent(loader, model):
    one = torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.])
    direction_r = torch.tensor([0.,-1.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    direction_l = torch.tensor([0.,0.,0.,0.,0.,0.,0.,-1.,0.,0.,0.,0.,0.,1.,0.])
    #direction = torch.randn(LATENT)
    eps = 0.1

    #initial_noise = torch.randn(LATENT)
    initial_noise = one

    fig = plt.figure(figsize=(1,LATENT_SAMPLES))

    for i in range(0, LATENT_SAMPLES):
        initial_noise[i] = 1
        noise = initial_noise.clone()
        for j in range(0, LATENT_SAMPLES):
            print(noise)
            res_image = model.decode(noise.reshape(1, -1))

            res_image = res_image.detach().numpy()
            fig.add_subplot(LATENT_SAMPLES, LATENT_SAMPLES, i * LATENT_SAMPLES + j + 1)
            plt.imshow(res_image[0].reshape(MNIST_X, MNIST_Y))

            noise += direction_l * eps

        initial_noise[i] = 0

    plt.show()


def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)

def load_model(model):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

