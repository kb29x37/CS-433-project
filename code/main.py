import load_data
import trainer
import models
import utils
import matplotlib.pyplot as plt
import matplotlib.image as image

from hyperparameters import *

train_loader, test_loader = load_data.load_MNIST_dataset()

#load_data.show_dataset_image(train_loader)

model = models.VAE_conv_mnist()
#model = models.VAE()

trainer.train_convnet(train_loader, model)
#trainer.train_fully_connected(train_loader, model)

print("done training")

utils.show_model_result_z(model, train_loader)

#utils.save_model(model)


