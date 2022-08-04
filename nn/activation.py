import numpy as np


def activation_function(act_f):
    if act_f == 'sigmoid':
        return Sigmoid()
    elif act_f == 'tanh':
        return Tanh()
    elif act_f == 'linear':
        return Linear()
    elif act_f == 'relu':
        return ReLU()


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


class Linear(Activation):
    def __init__(self):
        def linear(x):
            return x

        def linear_prime(x):
            return

        super().__init__(linear, linear_prime)


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return 0 if x < 0 else x

        def relu_prime(x):
            return 0 if x < 0 else 1

        super().__init__(relu, relu_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            # s = sigmoid(x)
            return x * (1 - x)

        super().__init__(sigmoid, sigmoid_prime)
