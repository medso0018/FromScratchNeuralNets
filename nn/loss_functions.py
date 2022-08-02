import numpy as np


def loss_function(loss_f='mse'):

    if loss_f == 'mse':
        return (lambda Yp, Yr: np.mean((Yp - Yr) ** 2), lambda Yp, Yr: Yp - Yr)
