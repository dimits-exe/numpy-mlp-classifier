import numpy as np
from common import sigmoid


class LogisticRegClassifier:
    """
    A binary classifier using logistic regression with regular gradient ascent (GDA).
    """

    def __init__(self, lamda: float, alpha: float, iters: int, print_history: bool = True):
        """
        Create a new logistic regression classification model.
        :param lamda: the regularization rate
        :param alpha: the learning rate
        :param iters: the number of iterations
        :param print_history: whether to print the error for every 100 iterations, use for debugging
        """
        self.lamda = lamda
        self.alpha = alpha
        self.iters = iters
        self.print_history = print_history
        self.weights = None

    def train(self, train_data: np.ndarray, train_labels) -> list[float]:
        """
        Train the model.
        :param train_data: a numpy array containing the training data
        :param train_labels: a binary numpy array containing the labels for the training data
        :return: a list of floats containing the error for every iteration during training
        """
        initial_weights = np.zeros((train_data.shape[1], 1))
        self.weights, cost_his = LogisticRegClassifier._gradient_ascent(train_data, train_labels, initial_weights,
                                                                        self.lamda, self.alpha, self.iters,
                                                                        self.print_history)
        return cost_his

    def test(self, test_data, test_labels) -> list[float]:
        """
        Test the model without modifying its internal weights.
        :param test_data: a numpy array containing the training data
        :param test_labels: a binary numpy array containing the labels for the training data
        :return:  a list of floats containing the error for every iteration during testing
        """
        return LogisticRegClassifier._gradient_ascent(test_data, test_labels, self.weights,
                                                      self.lamda, self.alpha, self.iters,
                                                      self.print_history)[1]

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the category for a set of data.
        :param data: a numpy array containing the data to be labeled
        :return: a numpy array containing the predicted label for each of the given data and a numpy array containing
        the probability that the ith label belongs to the second class
        """
        probability = sigmoid(data.dot(self.weights))
        category = probability > 0.5
        return category, probability

    @classmethod
    def _cost_gradient(cls, x: np.ndarray, y: np.ndarray, theta: np.ndarray, lamda: float) \
            -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the cost gradient
        :param x: a numpy array containing the data
        :param y: a numpy array containing the labels for the data
        :param theta: the current weights of the model
        :param lamda: the learning rate
        :return: a numpy array containing the cost for this iteration, and the computed gradient
        """
        h = sigmoid(x.dot(theta))
        regularization = (lamda / 2.0) * np.sum(theta ** 2)

        current_cost = (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h))) - regularization

        regularization = lamda * theta
        gradient = x.T.dot(y - h) / x.shape[0] - regularization
        return current_cost, gradient

    @classmethod
    def _gradient_ascent(cls, x: np.ndarray, y: np.ndarray, theta: np.ndarray, lamda: float, alpha: float, iters: int,
                         print_results: bool) -> tuple[np.ndarray, list[float]]:
        """
        Run the gradient ascent algorithm.
        :param x: a numpy array containing the data
        :param y: a numpy array containing the data's labels
        :param theta: the weight matrix of the model
        :param lamda: the regularization rate
        :param alpha: the learning rate
        :param iters: the number of iterations to run through
        :param print_results: whether to periodically print the cost while running
        :return:
        """
        cost_history = []

        for i in range(iters + 1):
            error, gradient = LogisticRegClassifier._cost_gradient(x, y, theta, lamda)

            cost_history.append(error[0])
            theta += alpha * gradient  # addition because we are ascending

            if print_results and i % 100 == 0:
                print("Iteration ", i, " Error:", error)

        return theta, cost_history
