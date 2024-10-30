def predict(network, x):
    y_pred = x
    for layer in network:
        y_pred = layer.forward(y_pred)
    return y_pred


def train(network, loss, loss_derivative, x_train, y_train, epochs=1000, learning_rate=0.01):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            output = predict(network, x)

            error += loss(y, output)

            grad = loss_derivative(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        error /= len(x_train)
        if e%10 == 0:
            print(f'Epoch {e+1} | Loss: {error}')
