import numpy as np
import torch

from hyperparameters import *

# compute k means on dataset
def k_means(loader):
    means_label = torch.zeros(K)

    (r_x, r_label) = next(iter(loader))
    r_x = r_x.resize(TEST_BATCH_SIZE, MNIST_IM_SIZE)
    r_x = r_x[torch.randperm(r_x.size(0))]

    # initialize mean array
    means = np.zeros((K, MNIST_IM_SIZE))
    step = int(r_x.size(0)/K)
    for i in range(0, K):
        means[i,:] = torch.mean(r_x[i*step:(i+1)*step,:], dim=0)
        print(means.shape)

    for e in range(0, K_MEANS_EPOCHS):
        for batch_idx, (x, target) in enumerate(loader):
            if(x.size()[0] == TEST_BATCH_SIZE):
                x = x.resize(TEST_BATCH_SIZE, MNIST_IM_SIZE).numpy()
                # find closest means

                indices = find_means(x, means)

                print("indices " + str(indices))
                print("target " + str(target.numpy()))

                for i in range(0, K):
                    selected_indices = np.asarray(np.where(indices == i))
                    if(selected_indices.size > 0):
                        means[i] = np.mean(x[selected_indices])

    return means

def compute_differences(x, means):
    indices = np.asarray([[i, j] for i in range(0, x.shape[0]) for j in range(0, K)])

    i_s = indices[:,0]
    j_s = indices[:,1]

    # compute the distance
    differences = np.square(x[i_s] - means[j_s])
    differences = differences.reshape(x.shape[0], K, MNIST_IM_SIZE)
    differences = np.sum(differences, axis=2)

    return differences

def find_means(batch, means):
    differences = compute_differences(batch, means)

    #print(differences.shape)

    return np.argmin(differences, axis=1)


def test_accuracy(loader, model, means):
    for batch_idx, (x, target) in enumerate(loader):
        if(x.size(0) == TEST_BATCH_SIZE):
            x = x.resize(TEST_BATCH_SIZE, MNIST_IM_SIZE)
            image_res, loss = model(x)

            image_res = image_res.detach().numpy()
            x = x.numpy()

            y_means = find_means(image_res, means)
            x_means = find_means(x, means)

            y = np.zeros(y_means.shape)
            for i in range(0, x_means.shape[0]):
                y[i] = 1 if y_means[i] == x_means[i] else 0

            acc = np.sum(y) / y.shape[0]

            print(acc)

            return y




