import numpy as np
from main.nn_components.layer import Layer


class Flatten(Layer):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, optim):
        return np.reshape(output_gradient, self.input_shape)