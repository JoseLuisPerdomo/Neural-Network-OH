import numpy as np


class Neuron(object):
    def __init__(self, inputs):
        self.inputs = [None] * inputs
        self.output = None
        self.weights = []
        for _ in range(inputs):
            self.weights.append(np.random.uniform())
        self.bias = np.random.uniform()

    def calculate_output(self):
        if len(self.inputs) != len(self.weights):
            raise ValueError(f"Dimensiones no coinciden: {len(self.inputs)} entradas y {len(self.weights)} pesos.")
        z = np.dot(self.weights, self.inputs) + self.bias
        self.output = 1 / (1 + np.exp(-z))
        return self.output
