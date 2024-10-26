from nn_components.neuron_class import Neuron
from nn_components.mse import mean_squared_error


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.network = []

        for i, layer_size in enumerate(layers):
            if i == 0:
                layer = [Neuron(layers[0]) for _ in range(layer_size)]  # Capas de entrada
            else:
                layer = [Neuron(layers[i - 1]) for _ in range(layer_size)]  # Capas ocultas y de salida
            self.network.append(layer)

    def forward_propagation(self, x, y_true):
        y_pred = []
        for sample in x:
            layer_output = sample

            for layer in self.network:
                next_input = []
                for neuron in layer:
                    neuron.inputs = layer_output  # Actualiza cada neurona con la salida de la capa previa
                    neuron_output = neuron.calculate_output()
                    next_input.append(neuron_output)

                layer_output = next_input

            y_pred.append(layer_output)

        return mean_squared_error(y_true, y_pred)




