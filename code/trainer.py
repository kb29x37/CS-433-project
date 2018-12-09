import numpy as np
import torch
import matplotlib.pyplot as plt

from hyperparameters import *

def train(train_set, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    for e in range(0, EPOCHS):
        print("Epoch: " + str(e))
        batch_losses = []
        for batch_idx, (x, target) in enumerate(train_set):
            if(x.size()[0] == BATCH_SIZE): # avoid last batch size potentially smaller
                y, mu, sigma = model(x.reshape(BATCH_SIZE,-1)) # compute predictions
                loss = model.ELBO_loss(mu, sigma, x, y)
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

