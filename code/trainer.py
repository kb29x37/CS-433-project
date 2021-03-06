import numpy as np
import torch
import matplotlib.pyplot as plt
import tester

from hyperparameters import *

def train_fully_connected(train_set, model, test_set):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    test_losses = []
    for e in range(0, EPOCHS):
        print("Epoch: " + str(e))
        batch_losses = []
        for batch_idx, (x, target) in enumerate(train_set):
            if(x.size()[0] == BATCH_SIZE): # avoid last batch size potentially smaller
                y, loss = model(x.reshape(BATCH_SIZE,-1)) # compute predictions
                print(loss.item())
                batch_losses.append(loss.item())

                # compute gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # compute batch losses after each epoch, to detect overfitting
        test_loss = tester.test_fully_connected(test_set, model)
        test_losses.append(np.mean(test_loss))

        losses.append(np.mean(batch_losses))
        #plt.plot(batch_losses)
        #plt.show()

    plt.plot(losses)
    plt.plot(test_losses)
    plt.show()

def train_convnet(train_set, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    for e in range(0, EPOCHS):
        print("Epoch: " + str(e))
        batch_losses = []
        for batch_idx, (x, target) in enumerate(train_set):
            if(x.size()[0] == BATCH_SIZE): # avoid last batch size potentially smaller
                loss, y_pred = model(x.reshape(BATCH_SIZE,1,MNIST_X,MNIST_Y)) # compute predictions
                print(loss)
                batch_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(np.mean(batch_losses))
        #plt.plot(batch_losses)
        #plt.show()

    plt.plot(losses)
    plt.show()

