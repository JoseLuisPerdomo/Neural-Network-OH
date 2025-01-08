import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


def dispersion_graph(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label='Test')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted Values')
    plt.legend()
    plt.show()


def learning_curve(loss_during_train):
    epochs = np.arange(len(loss_during_train))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_during_train, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()


def residual_errors_graph(y_true, y_pred):

    errors_train = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.hist(errors_train, bins=30, alpha=0.5, label='Train Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors')
    plt.legend()
    plt.show()


def plot_3d_points_and_plane_interactive(X, Y, coefficients):
    plt.ion()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    x1 = X[:, 0, 0]
    x2 = X[:, 1, 0]

    ax.scatter(x1, x2, Y.flatten(), c='blue', marker='o', label='Puntos de entrenamiento')

    # Crear una malla para el plano
    x1_range = np.linspace(x1.min(), x1.max(), 20)
    x2_range = np.linspace(x2.min(), x2.max(), 20)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    # Calcular el plano
    z_plane = coefficients[0] * x1_grid + coefficients[1] * x2_grid + coefficients[2]

    # Dibujar el plano
    ax.plot_surface(x1_grid, x2_grid, z_plane, alpha=0.5, color='orange')

    # Etiquetas
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title('Puntos de entrenamiento y plano ajustado (Interactivo)')

    # Mostrar leyenda
    ax.legend()

    # Mostrar gráfico interactivo
    plt.show(block=True)  # Bloquea la ejecución hasta que se cierre la ventana


def noisy_plane_points(n_points, n_dimensions, noise_level):
    np.random.seed(42)
    X = np.random.rand(n_points, n_dimensions) * 10
    coefficients = np.random.rand(n_dimensions)
    Y = X.dot(coefficients)

    Y_noisy = Y + noise_level * np.random.randn(n_points)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_noisy, test_size=0.2, random_state=42)

    X_train = np.reshape(X_train, (int(n_points*0.8), n_dimensions, 1))
    X_test = np.reshape(X_test, (int(n_points*0.2), n_dimensions, 1))
    Y_train = np.reshape(Y_train, (int(n_points*0.8), 1))
    Y_test = np.reshape(Y_test, (int(n_points*0.2), 1))

    return X_train, X_test, Y_train, Y_test, coefficients

