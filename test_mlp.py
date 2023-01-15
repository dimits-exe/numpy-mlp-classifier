import mlp
from load_mnist import load_data
from common import get_accuracy, sigmoid, sigmoid_prime, binary_x_entropy

import numpy as np
from typing import Callable
from unittest import TestCase


class ShallowNetworkTest(TestCase):
    """
    Test the accuracy and backpropagation algorithm used in the MLP classifier.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_data(42)

    def setUp(self) -> None:
        m = 25
        self.network = mlp.ShallowNetwork(input_size=784, hidden_size=m, output_size=1, eta=0.2, patience=5,
                                          tolerance=1e-6, activation_func=sigmoid, activation_func_prime=sigmoid_prime,
                                          cost_func_prime=binary_x_entropy)

    def test_accuracy(self):
        """
        Check whether the model successfully classifies examples over an arbitrary threshold.
        """
        self.network.gradient_descent(self.data.x_train, self.data.y_train)

        train_accuracy = get_accuracy(self.network.predict(self.data.x_train), self.data.y_train)
        assert train_accuracy > 0.7

        test_accuracy = get_accuracy(self.network.predict(self.data.x_test), self.data.y_test)
        assert test_accuracy > 0.7

    def test_back_prop(self):
        """
        Test whether the model's backpropagation algorithm produces a gradient close enough to a numerical
        approximation.
        """
        w_init = np.zeros((784, 25))

        grad_ew, numerical_grad = check_gradient(w_init, self.data.x_train, self.data.y_train,
                                                 lambda w, x, t: wrap_back_prop(w, x, t, self.network))
        diff = np.max(np.abs(grad_ew - numerical_grad))
        assert diff < 1  # TODO: lower this


def binary_x_entropy_prime(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (1 - y) / (1 - y_hat) - y / y_hat

def wrap_back_prop(w: np.ndarray, x_sample: np.ndarray, t_sample: np.ndarray, network: mlp.ShallowNetwork) \
        -> tuple[float, np.ndarray]:
    """
    Wrap a network instance to a function that is accepted by the provided check_gradient function.
    :param w: a numpy array containing the current weight
    :param x_sample: a numpy array containing the data points
    :param t_sample: a numpy array containing the labels for the data points
    :param network: the network instance to be wrapped
    :return: a numpy array containing the error and another containing the gradient
    """
    network.h_w = w
    results = network.back_propagation(x_sample, t_sample)
    grad = results[0]
    error = results[4][0][0]  # get the first element of the error array as a float
    return error, network.h_w - network.eta * grad


def check_gradient(w_init: np.ndarray, x: np.ndarray, t: np.ndarray, cost_grad: Callable) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    The provided function that checks the gradient's validity.
    :param w_init: a numpy array containing the initial weights
    :param x: a numpy array containing the data points
    :param t: a numpy array containing the labels for the data points
    :param cost_grad: a function that implements the back propagation procedure for one epoch
    :return: a numpy array containing the computed and another containing the numerical gradients
    """
    w = np.random.rand(*w_init.shape)
    epsilon = 1e-6

    _list = np.random.randint(x.shape[0], size=5)
    x_sample = np.array(x[_list, :])
    t_sample = np.array(t[_list, :])

    ew, grad_ew = cost_grad(w, x_sample, t_sample)

    numerical_grad = np.zeros(grad_ew.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in range(numerical_grad.shape[0]):
        for d in range(numerical_grad.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp = np.copy(w)
            w_tmp[k, d] += epsilon
            e_plus, _ = cost_grad(w_tmp, x_sample, t_sample)

            # subtract epsilon to the w[k,d]
            w_tmp = np.copy(w)
            w_tmp[k, d] -= epsilon
            e_minus, _ = cost_grad(w_tmp, x_sample, t_sample)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numerical_grad[k, d] = (e_plus - e_minus) / (2 * epsilon)
    return grad_ew, numerical_grad
