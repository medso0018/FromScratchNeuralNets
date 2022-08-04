
import numpy as np


def loss_function(loss_f='mse'):

    if loss_f == 'MSE':
        return MSE()


class Loss():

    def __init__(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime


class MSE(Loss):

    def __init__(self):
        def mse(y_true, y_pred):
            return np.mean(np.power(y_true - y_pred, 2))

        def mse_prime(y_true, y_pred):
            return 2 * (y_pred - y_true) / np.size(y_true)

        super().__init__(mse, mse_prime)
