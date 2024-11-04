import numpy as np
from main.nn_components.layer import Layer
from main.nn_components.optimazers import Adam, SGD


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-1, 1, (output_size, input_size))
        self.bias = np.random.uniform(-1, 1, (output_size, 1))

        # Adam
        self.v_weights = np.zeros_like(self.weights)
        self.s_weights = np.zeros_like(self.weights)
        self.v_bias = np.zeros_like(self.bias)
        self.s_bias = np.zeros_like(self.bias)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate, t, optim):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        if optim == 'adam':
            self.weights, self.bias, self.v_weights, self.s_weights, self.v_bias, self.s_bias = Adam(
                self.weights,
                self.bias,
                learning_rate,
                weights_gradient,
                output_gradient,
                self.v_weights,
                self.s_weights,
                self.v_bias,
                self.s_bias,
                gamma_v=0.9,
                gamma_s=0.999,
                epsilon=1e-8,
                t=t
            )
        elif optim == 'sgd':
            self.weights, self.bias = SGD(self.weights, self.bias, learning_rate, weights_gradient, output_gradient)
        return input_gradient
