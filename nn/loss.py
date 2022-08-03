
import numpy as np


def loss_function(loss_f='mse'):

    if loss_f == 'MSE':
        return MSE()
    elif loss_f == 'BCE':
        return BCE()


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


class BCE(Loss):

    def __init__(self):
        def bce(y_true, y_pred):
            return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

        def bce_prime(y_true, y_pred):
            return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

        super().__init__(bce, bce_prime)
