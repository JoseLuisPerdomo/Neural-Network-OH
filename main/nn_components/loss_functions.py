import numpy as np


class MSELoss:

    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)


class CrossEntropyLoss:

    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred))

    @staticmethod
    def derivative(y_true, y_pred):
        print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
        return y_pred - y_true
