from nn_components.neuron_class import Neuron


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.network = []

        for i, layer_size in enumerate(layers):
            if layer_size == 0:
                layer = [Neuron(1) for _ in range(layer_size)]
                self.network.append(layer)
                continue
            layer = [Neuron(layer[i - 1]) for _ in range(layer_size)]
            self.network.append(layer)
