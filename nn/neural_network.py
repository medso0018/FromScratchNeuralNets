from nn.layer import Layer
from nn.activation_functions import act_function
from nn.loss_functions import loss_function
import numpy as np


class NeuralNetwork():

    def __init__(self, loss_f='mse'):
        self.loss_f = loss_function(loss_f=loss_f)
        self.layers = []

    # add layer to our network
    def add_layer(self, n_neur=5, n_conn=None, act_f='linear'):
        _n_conn = n_conn if n_conn is not None else self.layers[-1].n_neur
        if len(self.layers) != 0:
            self.layers[-1].type = 'hidden_layer'
        self.layers.append(
            Layer(
                n_neur,
                n_conn=_n_conn,
                act_f=act_function(act_f)
            )
        )

    # train
    def fit(self, X_train, y_train, epoch=1000, lr=0.001):
        for _ in range(epoch):
            _output = self.__forward_propagation(X_train)
            _delta = self.__back_propagation(y_train, _output)
            self.__optimize(_output, _delta, lr, X_train, y_train)
            self.loss_f[0](_output[-1][1], y_train)
        return self

    # prediction
    def predict(self, X_test):
        return self.__forward_propagation(X_test)[-1][1]

    # forward propagation
    def __forward_propagation(self, X):
        output = [(None, X)]
        for layer in self.layers:
            z = output[-1][1] @ layer.W + layer.b
            a = layer.act_f[0](z)
            output.append((z, a))
        return output

    # backward propagation
    def __back_propagation(self, y, output):
        delta = []
        for l in reversed(range(0, len(self.layers))):
            _, a = output[l+1]
            if self.layers[l].type == 'output_layer':
                delta.insert(
                    0, self.loss_f[1](y[:, np.newaxis], a) * self.layers[l].act_f[1](a))
            else:
                delta.insert(
                    0, delta[0] @ self.layers[l+1].W.T * self.layers[l].act_f[1](a))
        return delta

    # optimization
    def __optimize(self, output, delta, lr, X, y):
        for l in reversed(range(0, len(self.layers))):
            _, a = output[l]
            self.layers[l].W = self.layers[l].W - a.T @ delta[l] * lr
            self.layers[l].b = self.layers[l].b - \
                np.mean(delta[l], axis=0, keepdims=True) * lr
