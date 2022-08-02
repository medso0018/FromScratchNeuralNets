from nn.layer import Layer
from nn.activation_functions import act_function
from nn.loss_functions import loss_function
import numpy as np


class NeuralNetwork():

    def __init__(self, loss_f='mse'):
        self.loss_f = loss_function(loss_f=loss_f)
        self.layers = []

    # add layer to our network
    def add_layer(self, n_neur=5, n_conn=None, act_f=None):
        if len(self.layers) == 0:
            n_conn = None
        else:
            n_conn = self.layers[-1].n_neur

        self.layers.append(
            Layer(
                n_neur,
                n_conn=n_conn,
                act_f=act_function(act_f)
            )
        )

    # train
    def fit(self, X, y, epoch=1000, lr=0.05):
        loss = []
        for _ in range(epoch):
            _output = self.__forward_propagation(X)
            _delta = self.__back_propagation(y, _output)
            self.__optimize(_output, _delta, lr)
            if epoch % 25 == 0:
                loss.append(self.loss_f[0](y, _output[-1]))
        return loss

    # prediction
    def predict(self, X_test):
        return self.__forward_propagation(X_test)[-1]

    # forward propagation
    def __forward_propagation(self, X):
        output = [X]
        for layer in self.layers[1:]:
            a = layer.act_f[0](output[-1] @ layer.W + layer.b)
            output.append(a)
        return output

    # backward propagation
    def __back_propagation(self, y, output):
        delta = []
        for l in reversed(range(0, len(self.layers))):
            a = output[l]
            if l != 0:
                if l == len(self.layers) - 1:
                    delta.insert(
                        0, self.loss_f[1](a, y) * self.layers[l-1].act_f[1](a))
                else:
                    delta.insert(
                        0, delta[0] @ self.layers[l+1].W.T * self.layers[l].act_f[1](a))
        return delta

    # optimization
    def __optimize(self, output, delta, lr):
        for l in reversed(range(0, len(self.layers) - 1)):
            a = output[l]
            self.layers[l+1].b = self.layers[l+1].b - \
                np.mean(delta[l], axis=0, keepdims=True) * lr
            self.layers[l+1].W = self.layers[l+1].W - a.T @ delta[l] * lr
