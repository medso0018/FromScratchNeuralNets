from nn.layer import Layer
from nn.loss import loss_function
import numpy as np


class NeuralNetwork():

    def __init__(self, loss_f='mse'):
        self.loss_f = loss_function(loss_f=loss_f)
        self.layers = []

    # add layer to the network
    def add_layer(self, layer):
        if len(self.layers) != 0:
            layer.connect(self.layers[-1])
        self.layers.append(layer)

    # train
    def fit(self, X, y, epoch=1000, lr=0.05):
        loss = []
        for _ in range(epoch):
            output = X
            for layer in self.layers[1:]:
                output = layer.forward(output)

            grade = self.loss_f.loss_prime(y, output) * \
                layer.act_f.activation_prime(output)

            for layer in self.layers[::-1][:-1]:
                grade = layer.backward(grade, lr)

            loss.append(self.loss_f.loss(y, output))

        return loss

    # prediction
    def predict(self, X):
        output = X
        for layer in self.layers[1:]:
            output = layer.forward(output)
        return output
