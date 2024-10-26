import numpy as np


class Neuron(object):
    def __init__(self, inputs):
        self.inputs = [None] * inputs
        self.last_input = 0
        self.output = None
        self.weights = []
        for _ in inputs:
            self.weights.append(np.random.uniform())
        self.bias = np.random.uniform()

    def connect_neuron(self, other_neuron):
        other_neuron.inputs[other_neuron.last_input] = self.output
        other_neuron.last_input += 1
