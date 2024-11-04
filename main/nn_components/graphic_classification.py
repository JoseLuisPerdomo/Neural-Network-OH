import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def display_confusion_matrix_iris(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Setosa', 'Versicolor', 'Virginica'])
    disp.plot()
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicciones')
    plt.ylabel('Verdaderos')
    plt.show()


def display_confusion_matrix_mnist(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    disp.plot()
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicciones')
    plt.ylabel('Verdaderos')
    plt.show()
