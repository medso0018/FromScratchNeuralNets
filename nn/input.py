import numpy as np
from nn.layer import Layer
from nn.activation import activation_function


class Input(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

    def forward(self, input):
        self.input = input
        return self.input
