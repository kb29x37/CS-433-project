import utils
import torch
import matplotlib.pyplot as plt
import numpy as np

from hyperparameters import *

def test_fully_connected(test_set, model):
    print("\n\n ================= TESTING ====================== \n\n")
    batch_losses = []
    NLL_losses = []
    for batch_idx, (x, target) in enumerate(test_set):
        if(x.size()[0] == BATCH_SIZE): # avoid last batch size potentially smaller
            y, loss = model(x.reshape(BATCH_SIZE,-1)) # compute predictions
            print(loss.item())
            batch_losses.append(loss.item())
            NLL_losses.append(torch.nn.functional.nll_loss(y, target).item())

        #plt.plot(batch_losses)
        #plt.show()

    print("loss on test_test: " + str(np.mean(batch_losses)))
    print("NLL estimation on test_test: " + str(np.mean(NLL_losses)))

    #utils.show_model_result_enc_dec(model, test_set)

    return batch_losses
