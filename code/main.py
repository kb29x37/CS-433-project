import load_data
import trainer
import tester
import models
import utils
import matplotlib.pyplot as plt
import matplotlib.image as image

from hyperparameters import *

train_loader, test_loader = load_data.load_MNIST_dataset()

model = utils.load_model(models.MAE_cleaned())

#load_data.show_dataset_image(train_loader)

#model = models.VAE_conv_mnist()
#model = models.MAE_conv_mnist()
#model = models.VAE()
#model = models.MAE_cleaned()

#trainer.train_convnet(train_loader, model)
#trainer.train_fully_connected(train_loader, model)

print("done training")
#tester.test_fully_connected(test_loader, model)

#utils.show_model_result_z_fully_connected(model, train_loader)

utils.linear_exploration_latent(test_loader, model)

#utils.show_model_result_z_convnet(model, train_loader)

#utils.save_model(model)



