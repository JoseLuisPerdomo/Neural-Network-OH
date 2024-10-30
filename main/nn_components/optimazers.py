def SGD(weights, bias, learning_rate, weights_gradient, output_gradient):
    weights = weights - learning_rate * weights_gradient
    bias = bias - learning_rate * output_gradient
    return weights, bias
