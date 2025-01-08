import numpy as np
from main.nn_components.layer import Layer


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-1, 1, (output_size, input_size))
        self.bias = np.random.uniform(-1, 1, (output_size, 1))

        self.optimizer.initialize(self.weights.shape, self.bias.shape)

    def forward(self, input):
        self.input = input
        return forward_with_batches(self.weights, self.input, self.bias)

    def backward(self, output_gradient, t, optim):

        weights_gradient = weight_gradient_with_batches(output_gradient, self.input)

        self.weights, self.bias = optim.update(
            self.weights, self.bias, weights_gradient, output_gradient
        )

        propagated_gradient = np.dot(self.weights.T, output_gradient)

        return propagated_gradient


def forward_with_batches(weights, inputs, bias):
    batch_size = inputs.shape[0]

    expanded_weights = np.expand_dims(weights, axis=0)
    expanded_weights = np.repeat(expanded_weights, batch_size, axis=0)
    expanded_bias = np.expand_dims(bias, axis=0)
    expanded_bias = np.repeat(expanded_bias, batch_size, axis=0)

    print(f"Expanded_weights: {expanded_weights.shape} with Input shape: {inputs.shape}")

    output = np.matmul(expanded_weights, inputs)

    print(f"Output (input * weights): {output.shape}")

    output += expanded_bias

    print(f"Output + bias: {output.shape}")

    return output


def weight_gradient_with_batches(output_gradient, inputs):

    print(f"Output gradient Shape: {output_gradient.shape}")

    print(f"Input Shape: {inputs.shape}")

    output = np.matmul(output_gradient, inputs.T)

    weights_gradient = np.mean(output, axis=0)

    return weights_gradient
