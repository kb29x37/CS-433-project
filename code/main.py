import load_data
import trainer
import tester
import models
import utils
#import k_means
import matplotlib.pyplot as plt
import matplotlib.image as image

from hyperparameters import *



train_loader, test_loader = load_data.load_MNIST_dataset()

#model = utils.load_model(models.MAE_cleaned())

#load_data.show_dataset_image(train_loader)

#model = models.VAE_conv_mnist()
#model = models.MAE_conv_mnist()
#model = models.VAE()
#model = models.MAE_cleaned()

#trainer.train_convnet(train_loader, model)
#trainer.train_fully_connected(train_loader, model, test_loader)

print("done training")

#tester.test_fully_connected(test_loader, model)

#utils.show_model_result_z_fully_connected(model, train_loader)

#utils.bilinear_exploration_latent(test_loader, model)

#utils.linear_exploration_latent(model,
#                                start=torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.]),
#                                direction=torch.tensor([0.,1.,0.,0.,0.,0.,0.,0.,-1.,0.,0.,0.,0.,1.,0.]))

#utils.show_model_result_z_convnet(model, train_loader)

#utils.show_model_result_enc_dec(model, test_loader)

#utils.save_model(model)

k_means.kmeans(test_loader)

