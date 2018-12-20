import numpy as np
import torch

from hyperparameters import *

# compute k means on dataset
def k_means(loader, batch_size, model):
    means_label = torch.zeros(K)

    (r_x, r_label) = next(iter(loader))

    mu, sigma = model.encode(r_x.resize(batch_size, MNIST_IM_SIZE))
    r_z = model.get_z(mu, sigma)

    r_z = r_z.resize(batch_size, LATENT)
    r_z = r_z[torch.randperm(r_z.size(0))]

    # initialize mean array
    means = np.zeros((K, LATENT))
    step = int(r_z.size(0)/K)
    for i in range(0, K):
        means[i,:] = torch.mean(r_z[i*step:(i+1)*step,:].detach(), dim=0)
        print(means.shape)

    for e in range(0, K_MEANS_EPOCHS):
        for batch_idx, (x, target) in enumerate(loader):
            if(x.size()[0] == batch_size):
                x = x.resize(batch_size, MNIST_IM_SIZE)
                # find closest means

                mu, sigma = model.encode(x)
                z = model.get_z(mu, sigma)

                z = z.detach().numpy()
                x = x.numpy()

                indices = find_means(z, means)

                #print("indices " + str(indices))
                #print("target " + str(target.numpy()))

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
    differences = differences.reshape(x.shape[0], K, LATENT)
    differences = np.sum(differences, axis=2)

    return differences

def find_means(batch, means):
    differences = compute_differences(batch, means)

    #print(differences.shape)

    return np.argmin(differences, axis=1)

def test_accuracy(test_loader, model, train_loader):
    means = k_means(train_loader, BATCH_SIZE, model)

    means_labeled = np.zeros(K)
    closest_distance = np.zeros(LATENT)

    for batch_idx, (x, target) in enumerate(train_loader):
        if(x.size()[0] == batch_size):
            mu, sigma = model.encode(x.resize(BATCH_SIZE, MNIST_IM_SIZE))
            z = model.get_z(mu, sigma)

            differences = compute_differences(z, means)
            closest_means = np.argmin(differences, axis=1)

            for i in range(0, BATCH_SIZE):
                if(closest_distance[closest_means[i]] > differences[i, closest_means[i]]):
                    closest_distance[closest_means[i]] = differences[i, closest_means[i]]
                    means_labeled[closest_means[i]] = target[i]


    acc = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        if(x.size()[0] == batch_size):
            mu, sigma = model.encode(x.resize(TEST_BATCH_SIZE, MNIST_IM_SIZE))
            z = model.get_z(mu, sigma)

            y_means = find_means(z, means)

            for i in range(0, y_means.shape[0]):
                if(means_labeled[y_means[i]] == target[i]):
                    acc += 1

    acc = acc / len(test_loader.dataset)

    return acc




