import numpy as np


class LogisticRegClassifier:

    def __init__(self, lamda: float, alpha: float, iters: int, print_history: bool = True):
        self.lamda = lamda
        self.alpha = alpha
        self.iters = iters
        self.print_history = print_history
        self.weights = None

    def train(self, train_data, train_labels) -> list[float]:
        initial_weights = np.zeroes(train_data.shape[1]).reshape((-1, 1))
        self.weights, cost_his = LogisticRegClassifier._gradient_ascent(train_data, train_labels, initial_weights,
                                                                        self.lamda, self.alpha, self.iters,
                                                                        self.print_history)
        return cost_his

    def test(self, test_data, test_labels) -> list[float]:
        return LogisticRegClassifier._gradient_ascent(test_data, test_labels, self.weights,
                                                      self.lamda, self.alpha, self.iters,
                                                      self.print_history)[1]

    def predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probability = sigmoid(data.dot(self.weights))
        category = probability > 0.5
        return category, probability

    @classmethod
    def _cost_gradient(cls, x: np.ndarray, y: np.ndarray, theta: np.ndarray, lamda: float) \
            -> tuple[np.ndarray, np.ndarray]:
        m = float(x.shape[0])

        h = sigmoid(x.dot(theta))
        regularization = 1 / (2 * m) * lamda * np.sum(theta ** 2)
        # why did we apply a '-' here?
        current_j = (1.0 / m) * ((-y).dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + regularization

        regularization = lamda * theta / m  # why do we divide by m?
        gradient = (1.0 / m) * x.T.dot(h - y) + regularization
        return current_j, gradient

    @classmethod
    def _gradient_ascent(cls, x: np.ndarray, y: np.ndarray, theta: np.ndarray, lamda: float, alpha: float, iters: int,
                         print_results: bool) -> tuple[np.ndarray, list[float]]:
        cost_history = []

        for i in range(iters):
            error, gradient = LogisticRegClassifier._cost_gradient(x, y, theta, lamda)
            theta += alpha * gradient  # addition because we are ascending

            if print_results:
                print(error)
            cost_history.append(error[0])

        return theta, cost_history


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(x))
