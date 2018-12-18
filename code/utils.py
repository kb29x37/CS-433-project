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

    res_image, mu, logvar = model(image.reshape(1, -1))

    fig = plt.figure(figsize=(2,1))

    res_image = res_image.detach().numpy()
    fig.add_subplot(2, 1, 0)
    plt.imshow(res_image[0].reshape(MNIST_X, MNIST_Y))
    fig.add_subploy(2, 1, 1)
    plt.imshow(image)
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


def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)

def load_model():
    model = VAE()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

