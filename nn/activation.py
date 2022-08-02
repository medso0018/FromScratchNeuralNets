import numpy as np


def activation_function(act_f):
    if act_f == 'sigmoid':
        return Sigmoid()
    elif act_f == 'tanh':
        return Tanh()


class Activation():
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            # s = sigmoid(x)
            return x * (1 - x)

        super().__init__(sigmoid, sigmoid_prime)
