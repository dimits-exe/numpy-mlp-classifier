from typing import Callable
import numpy as np


class ShallowNetwork:
    """
    A binary MLP classifier with one hidden layer.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, eta: float, patience: int,
                 tolerance: float, activation_func: Callable[[np.ndarray], np.ndarray],
                 activation_func_prime: Callable[[np.ndarray], np.ndarray],
                 cost_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 cost_func_prime: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """
        Initialize the parameters of the model.
        :param input_size: the number of input neurons in the network
        :param hidden_size: the number of hidden neurons in the network
        :param output_size: the number of output neurons in the network
        :param eta: the learning rate
        :param patience: the threshold for early stopping
        :param tolerance: minimum change in the monitored quantity to qualify as an improvement,
         i.e. an absolute change of less than the tolerance, will count as no improvement.
        :param activation_func: the activation function
        :param activation_func_prime: the derivative of the activation function
        :param cost_func_prime: the derivative of the cost function
        """
        self.input_size = input_size
        self.eta = eta
        self.patience = patience
        self.tolerance = tolerance
        self.activation_func = activation_func
        self.activation_func_prime = activation_func_prime
        self.cost_func = cost_func
        self.cost_func_prime = cost_func_prime

        # don't set weights and biases for the input
        self.h_b = np.zeros((1, hidden_size))
        self.o_b = np.zeros((1, output_size))

        self.h_w = np.zeros((input_size, hidden_size))
        self.o_w = np.zeros((hidden_size, output_size))

    def train(self, train_data: np.ndarray, train_labels: np.ndarray, val_data: np.ndarray,
              val_labels: np.ndarray) -> tuple[int, float, list[float]]:
        """
        Train the network on a set of training data.
        :param train_data: a numpy array containing the train data
        :param train_labels: a numpy array containing the respective labels for the training data
        :param val_data: a numpy array containing the validation data used for the early stopping algorithm
        :param val_labels: a numpy array containing the labels for the validation data used for the early stopping
        algorithm
        :return: the number of epochs used, the minimum error found, and the error history
        """
        grad, epoch, last_error, error_history = self._gradient_descent(train_data, train_labels, val_data, val_labels)
        self.h_w = grad["h_w"]
        self.o_w = grad["o_w"]
        self.h_b = grad["h_b"]
        self.o_b = grad["o_b"]

        return epoch, last_error, error_history

    def _gradient_descent(self, train_data: np.ndarray, train_labels: np.ndarray, val_data: np.ndarray,
                          val_labels: np.ndarray) -> tuple[dict[str, np.ndarray], int, float, list[float]]:
        """
        Implements the entire gradient descent procedure, calculating the gradient and loss.
        :param train_data: a numpy array containing the train data
        :param train_labels: a numpy array containing the respective labels for the training data
        :param val_data: a numpy array containing the validation data used for the early stopping algorithm
        :param val_labels: a numpy array containing the labels for the validation data used for the early stopping
        algorithm
        :return: a dictionary containing the gradient, the number of epochs used, the minimum error found, and the error
        history
        """
        epoch: int = 0  # logging
        least_error: float = np.inf
        epochs_since_improvement: int = 0
        best_model_params = {}
        error_history = []

        h_w = self.h_w.copy()
        o_w = self.o_w.copy()
        h_b = self.h_b.copy()
        o_b = self.o_b.copy()

        while epochs_since_improvement <= self.patience:
            parameters = {"h_w": h_w, "o_w": o_w, "h_b": h_b, "o_b": o_b}
            dw1, dw2, db1, db2, train_loss = self._back_propagation(train_data, train_labels, parameters)
            error_history.append(train_loss)

            h_w -= self.eta * dw1
            o_w -= self.eta * dw2
            h_b -= self.eta * db1
            o_b -= self.eta * db2

            # early stopping
            val_loss: float = self._forward_pass(parameters, val_data, val_labels)[4]
            if val_loss + self.tolerance < least_error:
                least_error = val_loss
                epochs_since_improvement = 0
                best_model_params: dict[str, np.ndarray] = parameters
            else:
                epochs_since_improvement += 1

            epoch += 1

        return best_model_params, epoch, least_error, error_history

    def _back_propagation(self, x: np.ndarray, y: np.ndarray, parameters: dict[str, np.ndarray]) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Implements the back-propagation procedure for one epoch.
        :param parameters: a dictionary containing the last computed weights and biases
        :param x: a numpy array containing the train data
        :param y: a numpy array containing the respective labels for the training data
        :return: a tuple containing the gradients for the hidden weights, output weights, hidden bias, output bias and
        the current error respectively
        """
        a1, z1, a2, z2, error = self._forward_pass(parameters, x, y)

        # Backward pass
        m = x.shape[1]

        # output layer activation derivative
        dy_hat = self.cost_func_prime(a2, y)
        dz2 = dy_hat * self.activation_func_prime(z2)
        # something is wrong in the line below, and it fucks the computation under specific circumstances
        dw2: np.ndarray = (1 / m) * a1.T.dot(dz2)
        db2: np.ndarray = (1 / m) * np.sum(dz2, axis=0)

        # hidden layer activation derivative
        da1 = dz2.dot(parameters["o_w"].T)
        dz1 = da1 * self.activation_func_prime(z1)
        dw1: np.ndarray = (1 / m) * x.T.dot(dz1)
        db1: np.ndarray = (1 / m) * np.sum(dz1, axis=0)

        return dw1, dw2, db1, db2, error

    def _forward_pass(self, parameters: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray = None) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Implement a forward pass through the network and get the resulting outputs and error
        :param parameters: a dictionary containing the last computed weights and biases
        :param x: a numpy array containing the data
        :param y: a numpy array containing the respective labels for the data, None if the method is being called for
        prediction (and thus no labels are provided)
        :return: a tuple containing the logits and output for the hidden layer and output layers respectively, as well
        as the mean error if a y array was provided

        """
        h_w = parameters["h_w"]
        o_w = parameters["o_w"]
        h_b = parameters["h_b"]
        o_b = parameters["o_b"]

        # Forward pass
        z1 = x.dot(h_w) + h_b
        a1 = self.activation_func(z1)
        z2 = a1.dot(o_w) + o_b
        a2 = self.activation_func(z2)

        if y is not None:
            error = self.cost_func(a2, y).mean()
        else:
            error = None

        return a1, z1, a2, z2, error

    def output(self, x: np.ndarray) -> np.ndarray:
        """
        Return the results produced by the model for the provided data array.
        :param x: a numpy array containing the data to be classified
        :return: a numpy array containing the logit for each data point
        """
        parameters = {"h_w": self.h_w, "o_w": self.o_w, "h_b": self.h_b, "o_b": self.o_b}
        return self._forward_pass(parameters, x)[2]

    def predict(self, test_data: np.ndarray, test_labels: np.ndarray = None) -> tuple[np.ndarray, float]:
        """
        Get the predicted classification for the provided data array.
        :param test_data: a numpy array containing the data to be classified
        :param test_labels: a numpy array containing the data to be classified
        :return: a numpy array containing a binary classification for each data point and the mean error
        """
        parameters = {"h_w": self.h_w, "o_w": self.o_w, "h_b": self.h_b, "o_b": self.o_b}
        _, _, output, _, loss = self._forward_pass(parameters, test_data, test_labels)
        return np.where(output < 0.5, 0, 1), loss
