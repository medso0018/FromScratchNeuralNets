import numpy as np
from scipy import signal
from nn.layer import Layer
from nn.activation import activation_function


class Convolutional(Layer):

    def __init__(self, kernel_size, depth, act_f='sigmoid'):
        self.depth = depth
        self.kernel_size = kernel_size
        self.act_f = activation_function(act_f)

    def connect(self, other_layer):
        self.input_depth, input_height, input_width = other_layer.output_shape
        self.kernels_shape = (self.depth, self.input_depth,
                              self.kernel_size, self.kernel_size)
        self.input_shape = other_layer.output_shape
        self.output_shape = (
            self.depth,
            input_height - self.kernel_size + 1,
            input_width - self.kernel_size + 1
        )
        self.K = np.random.rand(*self.kernels_shape) * 2 - 1
        self.b = np.random.rand(*self.output_shape) * 2 - 1

    def forward(self, input):
        self.input = input
        self.output = np.zeros(self.output_shape)
        for d in range(self.depth):
            for i_d in range(self.input_depth):
                b = self.b[d]
                kx = signal.correlate2d(
                    self.input[i_d],  self.K[d, i_d], 'valid')
                self.output[d] = self.act_f.activation(kx + b)
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernel_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for d in range(self.depth):
            for i_d in range(self.input_depth):
                kernel_gradient[d, i_d] += signal.correlate2d(
                    self.input[i_d], output_gradient[d], 'valid')

                input_gradient[i_d] += signal.correlate2d(
                    output_gradient[d], self.K[d, i_d], 'full') * self.act_f.activation_prime(self.input[i_d])

        self.K -= learning_rate * kernel_gradient
        self.b -= learning_rate * output_gradient

        return input_gradient
