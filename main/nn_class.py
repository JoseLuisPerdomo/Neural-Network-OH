from nn_components import draw_nn
from nn_components.neuron_class import Neuron


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.network = []

        for layer_size in layers:
            layer = [Neuron(3) for _ in range(layer_size)]
            self.network.append(layer)

    @staticmethod
    def draw():
        return draw_nn
