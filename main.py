from nn.neural_network import NeuralNetwork

import pandas as pd
import numpy as np

from nn.dense import Dense
from nn.input import Input
from nn.convolutional import Convolutional
from nn.reshap import Reshape


x = np.array([
    np.random.rand(100, 100),
])


model = NeuralNetwork(loss_f='MSE')

model.add_layer(Input(input_shape=x.shape))
model.add_layer(Convolutional(kernel_size=3, depth=2, act_f='sigmoid'))
model.add_layer(Reshape())
model.add_layer(Dense(n_neur=4, act_f='sigmoid'))
model.add_layer(Dense(n_neur=3, act_f='sigmoid'))
