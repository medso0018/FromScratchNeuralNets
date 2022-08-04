import numpy as np
from nn.layer import Layer
from nn.activation import activation_function


class Dense(Layer):
    def __init__(self, n_neur, act_f='sigmoid'):
        self.act_f = activation_function(act_f)
        self.n_neur = n_neur
        self.output_shape = (n_neur, 1)

        self.b = np.random.rand(n_neur, 1) * 2 - 1
        self.W = None

    def connect(self, other_layer):
        self.W = np.random.rand(
            self.n_neur, other_layer.output_shape[0]) * 2 - 1

    def forward(self, input):
        self.input = input
        return self.act_f.activation(self.W @ self.input + self.b)

    def backward(self, output_gradient, learning_rate):
        weights_gradient = output_gradient @ self.input.T
        input_gradient = self.W.T @ output_gradient * \
            self.act_f.activation_prime(self.input)

        self.W -= learning_rate * weights_gradient
        self.b -= learning_rate * output_gradient
        return input_gradient
