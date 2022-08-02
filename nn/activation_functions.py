import numpy as np


def act_function(act_f='sigmoid'):

    if act_f == 'sigmoid':
        return (lambda x: 1 / (1 + np.exp(-x)), lambda x: x * (1 - x))
