from nn.neural_network import NeuralNetwork
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from nn.layers import Input, Convolutional, Dense, Reshape

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils


def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y


# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)


model = NeuralNetwork(loss_f='MSE')

model.add_layer(Input(input_shape=x_train.shape[1:]))
model.add_layer(Convolutional(kernel_size=3, depth=2, act_f='sigmoid'))
model.add_layer(Convolutional(kernel_size=3, depth=1, act_f='sigmoid'))
model.add_layer(Reshape())
model.add_layer(Dense(n_neur=4, act_f='sigmoid'))
model.add_layer(Dense(n_neur=2, act_f='sigmoid'))


for layer in model.layers:
    print(layer.output_shape)

loss = model.fit(x_train, y_train, epochs=300, lr=0.005)
plt.figure(figsize=(10, 6))
plt.plot(list(range(len(loss))), loss)
plt.title('Loss curve')
plt.show()
