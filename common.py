import numpy as np


def get_accuracy(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> float:
    """
    Get the accuracy of the model based on its predicted and actual labels of its data.
    :param predicted_labels: the labels which the model predicted
    :param actual_labels: the actual labels of the data
    :return: a number between 0 and 1 representing the accuracy of the model
    """
    true_predictions = np.count_nonzero(np.where(predicted_labels == 0, 0, 1) == actual_labels)
    return true_predictions / actual_labels.shape[0]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def binary_x_entropy_prime(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (1 - y) / (1 - y_hat) - y / y_hat
