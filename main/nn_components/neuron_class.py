import numpy as np


class Neuron(object):
    def __init__(self, inputs):
        self.inputs = [None] * inputs
        self.output = None
        self.weights = []
        for _ in inputs:
            self.weights.append(np.random.uniform())
        self.bias = np.random.uniform()

    def connect_neuron(self, other_neuron):
        pass

    def set_weights(self, weights):
        self.weights = weights
