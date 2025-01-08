import numpy as np
from main.nn_components.activation_layer import Activation


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x):
            sigmoid_x = sigmoid(x)
            return sigmoid_x * (1 - sigmoid_x)

        super().__init__(sigmoid, sigmoid_derivative)


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_derivative(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_derivative)


class Relu(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_derivative(x):
            return np.where(x > 0, 1, 0)

        super().__init__(relu, relu_derivative)


class LeakyRelu(Activation):
    def __init__(self, alpha=0.01):
        def leaky_relu(x):
            return np.where(x > 0, x, alpha * x)

        def leaky_relu_derivative(x):
            return np.where(x > 0, 1, alpha)

        super().__init__(leaky_relu, leaky_relu_derivative)
