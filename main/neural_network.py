import matplotlib.pyplot as plt
from IPython.display import clear_output


def predict(network, x):
    y_pred = x
    for layer in network:
        y_pred = layer.forward(y_pred)
    return y_pred


def train(network, loss, loss_derivative, x_train, y_train, optim, epochs=1000, learning_rate=0.01):
    history = []
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            output = predict(network, x)

            error += loss(y, output)

            grad = loss_derivative(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate, e, optim)
        error /= len(x_train)
        history.append(error)
        if e % 100 == 0:
            print(f"Epoch: {e}, Loss: {error}, % of epochs: {round(e/epochs, 2) * 100}%")
        clear_output(wait=True)
    print(f"Final Loss = {error}")

    return history
