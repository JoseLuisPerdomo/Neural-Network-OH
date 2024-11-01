import numpy as np


def SGD(weights, bias, learning_rate, weights_gradient, output_gradient):
    weights = weights - learning_rate * weights_gradient
    bias = bias - learning_rate * output_gradient
    return weights, bias


def Adam(weights, bias, learning_rate, weights_gradient, output_gradient,
         v_weights, s_weights, v_bias, s_bias,
         gamma_v=0.9, gamma_s=0.999, epsilon=1e-8, t=2):

    v_weights = gamma_v * v_weights + (1 - gamma_v) * weights_gradient
    s_weights = gamma_s * s_weights + (1 - gamma_s) * (weights_gradient ** 2)

    v_bias = gamma_v * v_bias + (1 - gamma_v) * output_gradient
    s_bias = gamma_s * s_bias + (1 - gamma_s) * (output_gradient ** 2)

    v_weights_corrected = v_weights / (1 - gamma_v ** t if (1 - gamma_v ** t) > epsilon else epsilon)
    s_weights_corrected = s_weights / (1 - gamma_s ** t if (1 - gamma_s ** t) > epsilon else epsilon)
    v_bias_corrected = v_bias / (1 - gamma_v ** t if (1 - gamma_v ** t) > epsilon else epsilon)
    s_bias_corrected = s_bias / (1 - gamma_s ** t if (1 - gamma_s ** t) > epsilon else epsilon)

    weights = weights - learning_rate * v_weights_corrected / (np.sqrt(s_weights_corrected) + epsilon)
    bias = bias - learning_rate * v_bias_corrected / (np.sqrt(s_bias_corrected) + epsilon)

    return weights, bias, v_weights, s_weights, v_bias, s_bias
