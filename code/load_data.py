import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from hyperparameters import *

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def load_MNIST_dataset():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (MNIST_MEAN,), (MNIST_STD,))
                                   ])),
        batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (MNIST_MEAN,), (MNIST_STD,))
                                   ])),
        batch_size=TEST_BATCH_SIZE, shuffle=True)

    return train_loader, test_loader

def load_CIFAR10_dataset():
    transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def show_dataset_image(loader):
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    plt.imshow(example_data[0][0])
    plt.show()

    print(example_data[0][0])

def get_dataset_image(loader):
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    return example_data[0][0]

