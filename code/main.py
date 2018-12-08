import load_data
import trainer
import models
import utils
import matplotlib.pyplot as plt
import matplotlib.image as image

from hyperparameters import *

train_loader, test_loader = load_data.load_MNIST_dataset()

#load_data.show_dataset_image(train_loader)

model = models.VAE()

trainer.train(train_loader, model)

print("done training")

utils.show_model_result_enc_dec(model, train_loader)

