from typing import Callable
import numpy as np


class ShallowNetwork:
    """
    A binary MLP classifier with one hidden layer.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, eta: float, patience: int,
                 activation_func: Callable[[np.ndarray], np.ndarray],
                 activation_func_prime: Callable[[np.ndarray], np.ndarray],
                 cost_func_prime: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """
        Initialize the parameters of the model.
        :param input_size: the number of input neurons in the network
        :param hidden_size: the number of hidden neurons in the network
        :param output_size: the number of output neurons in the network
        :param eta: the learning rate
        :param patience: the threshold for early stopping
        :param activation_func: the activation function
        :param activation_func_prime: the derivative of the activation function
        :param cost_func_prime: the derivative of the cost function
        """
        self.input_size = input_size
        self.eta = eta
        self.patience = patience
        self.activation_func = activation_func
        self.activation_func_prime = activation_func_prime
        self.cost_func_prime = cost_func_prime

        # don't set weights and biases for the input
        self.h_b = np.zeros((1, hidden_size))
        self.o_b = np.zeros((1, output_size))

        self.h_w = np.zeros((input_size, hidden_size))
        self.o_w = np.zeros((hidden_size, output_size))

    def gradient_descent(self, train_data: np.ndarray, train_labels: np.ndarray) -> tuple[int, float, list[float]]:
        """
        Implements the entire gradient descent procedure, updating the model's internal weights and biases.
        :param train_data: a numpy array containing the train data
        :param train_labels: a numpy array containing the respective labels for the training data
        :return: the number of epochs needed to reach the early stopping point, the minimum error found, and the error
        history
        """
        epoch: int = 0  # logging
        least_error: float = np.inf
        epochs_since_improvement: int = 0
        best_model_params = None
        error_history = []

        while epochs_since_improvement <= self.patience:
            dw1, dw2, db1, db2, cost = self.back_propagation(train_data, train_labels)

            self.h_w -= self.eta * dw1
            self.o_w -= self.eta * dw2
            self.h_b -= self.eta * db1
            self.o_b -= self.eta * db2

            # early stopping
            error = abs(cost.mean())
            if error < least_error:
                print(f"Iteration {epoch} improvement from {least_error} to {error}")
                least_error = error
                epochs_since_improvement = 0
                best_model_params = self.h_w, self.o_w, self.h_b, self.o_b
            else:
                epochs_since_improvement += 1
                print(
                    f"Iteration {epoch} NO improvement from {least_error} to {error}, "
                    f"increasing to{epochs_since_improvement}")

            error_history.append(error)
            epoch += 1

        # keep best model params
        self.h_w = best_model_params[0]
        self.o_w = best_model_params[1]
        self.h_b = best_model_params[2]
        self.o_b = best_model_params[3]

        return epoch, least_error, error_history

    def back_propagation(self, x: np.ndarray, y: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Implements the back-propagation procedure for one epoch.
        :param x: a numpy array containing the train data
        :param y: a numpy array containing the respective labels for the training data
        :return: a tuple containing the gradients for the hidden weights, output weights, hidden bias, output bias and
        the current error respectively
        """

        # Forward pass
        z1 = x.dot(self.h_w) + self.h_b
        a1 = self.activation_func(z1)
        z2 = a1.dot(self.o_w) + self.o_b
        a2 = self.activation_func(z2)

        # Backward pass
        m = x.shape[1]

        # output layer activation derivative
        dy_hat = self.cost_func_prime(a2, y)
        dz2 = dy_hat * self.activation_func_prime(z2)
        dw2: np.ndarray = (1 / m) * a1.T.dot(dz2)
        db2: np.ndarray = (1 / m) * np.sum(dz2, axis=0)

        # hidden layer activation derivative
        da1 = dz2.dot(self.o_w.T)
        dz1 = da1 * self.activation_func_prime(z1)
        dw1: np.ndarray = (1 / m) * x.T.dot(dz1)
        db1: np.ndarray = (1 / m) * np.sum(dz1, axis=0)

        return dw1, dw2, db1, db2, dz2

    def output(self, x: np.ndarray) -> np.ndarray:
        """
        Return the logits produced by the model for the provided data array.
        :param x: a numpy array containing the data to be classified
        :return: a numpy array containing the logit for each data point
        """
        # pass to hidden layer
        x = self.activation_func(x.dot(self.h_w) + self.h_b)
        # pass to output layer
        x = self.activation_func(x.dot(self.o_w) + self.o_b)
        return x

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Get the predicted classification for the provided data array.
        :param test_data: a numpy array containing the data to be classified
        :return: a numpy array containing a binary classification for each data point
        """
        return np.where(self.output(test_data) < 0.5, 0, 1)
