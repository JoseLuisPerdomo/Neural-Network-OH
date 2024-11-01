import matplotlib.pyplot as plt
from IPython.display import clear_output

clear_output(wait=True)



def predict(network, x):
    y_pred = x
    for layer in network:
        y_pred = layer.forward(y_pred)
    return y_pred


def train(network, loss, loss_derivative, x_train, y_train, optim, graph=True, epochs=1000, learning_rate=0.01):
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
        print(f"Epoch: {e}, Error: {error}")

    if graph:

        plt.plot(history[1:])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss during training")
        plt.show()
