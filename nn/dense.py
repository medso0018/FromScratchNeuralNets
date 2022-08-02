import numpy as np
from nn.layer import Layer
from nn.activation import activation_function


class Dense(Layer):
    def __init__(self, n_neur, act_f='sigmoid'):
        self.act_f = activation_function(act_f)
        self.n_neur = n_neur
        self.output_size = n_neur

        self.b = np.random.rand(1, n_neur) * 2 - 1
        self.W = None

    def connect(self, other_layer):
        self.W = np.random.rand(other_layer.output_size, self.n_neur) * 2 - 1

    def forward(self, input):
        self.input = input
        return self.act_f.activation(self.input @ self.W + self.b)

    def backward(self, output_gradient, learning_rate):
        weights_gradient = self.input.T @ output_gradient
        input_gradient = output_gradient @ self.W.T * \
            self.act_f.activation_prime(self.input)
        self.W -= learning_rate * weights_gradient
        self.b -= learning_rate * \
            np.mean(output_gradient, axis=0, keepdims=True)
        return input_gradient
