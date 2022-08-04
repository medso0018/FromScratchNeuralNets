from nn.layer import Layer
from nn.loss import loss_function
import numpy as np


class NeuralNetwork():

    def __init__(self, loss_f='MSE'):
        self.loss_f = loss_function(loss_f=loss_f)
        self.layers = []

    # add layer to the network
    def add_layer(self, layer):
        if len(self.layers) != 0:
            layer.connect(self.layers[-1])
        self.layers.append(layer)

    # train
    def fit(self, X, Y, epochs=1000, lr=0.05):
        loss = []
        for _ in range(epochs):
            error = 0
            for x, y in zip(X, Y):

                # forward
                out = x
                for layer in self.layers[1:]:
                    out = layer.forward(out)

                # error
                error += self.loss_f.loss(y, out)

                # backward
                grad = self.loss_f.loss_prime(y, out)
                for layer in self.layers[::-1][:-1]:
                    grad = layer.backward(grad, lr)

            loss.append(error / len(Y))
        return loss

    # prediction

    def predict(self, X):
        output = X
        for layer in self.layers[1:]:
            output = layer.forward(output)
        return output
