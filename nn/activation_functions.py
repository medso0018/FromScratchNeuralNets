import numpy as np


def act_function(act_f='linear'):

    if act_f == 'linear':
        return (lambda x: x, lambda _: 1)
        return
    if act_f == 'relu':
        return (lambda x: np.max([0, x]), lambda x: 1 if x > 0 else 0)
    elif act_f == 'sigmoid':
        return (lambda x: 1 / (1 + np.exp(-x)), lambda x: x * (1 - x))
