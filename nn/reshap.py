import numpy as np
from nn.layer import Layer
from nn.activation import activation_function


class Reshape(Layer):

    def connect(self, other_layer):
        depth, height, width = other_layer.output_shape
        self.output_shape = (depth * height * width, 1)

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, _):
        return np.reshape(output_gradient, self.input_shape)
