import numpy as np
from nn.activation import activation_function
from scipy import signal


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass


class Input(Layer):
    def __init__(self, input_shape):
        self.input_shape = 'None'
        self.output_shape = input_shape
        self.act_f_name = 'None'

    def forward(self, input):
        self.input = input
        return self.input


class Dense(Layer):
    def __init__(self, n_neur, act_f='sigmoid'):
        self.act_f = activation_function(act_f)
        self.n_neur = n_neur
        self.output_shape = (n_neur, 1)
        self.act_f_name = act_f.capitalize()

        self.b = np.random.rand(n_neur, 1) * 2 - 1
        self.W = None

    def connect(self, other_layer):
        self.W = np.random.rand(
            self.n_neur, other_layer.output_shape[0]) * 2 - 1
        self.input_shape = other_layer.output_shape

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


class Reshape(Layer):

    def connect(self, other_layer):
        depth, height, width = other_layer.output_shape
        self.input_shape = other_layer.output_shape
        self.output_shape = (depth * height * width, 1)
        self.act_f_name = 'None'

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, _):
        return np.reshape(output_gradient, self.input_shape)


class Convolutional(Layer):

    def __init__(self, kernel_size, depth, act_f='sigmoid'):
        self.depth = depth
        self.kernel_size = kernel_size
        self.act_f = activation_function(act_f)
        self.act_f_name = act_f.capitalize()

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
