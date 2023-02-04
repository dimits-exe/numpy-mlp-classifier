from typing import Callable

from models import mlp
from load_mnist import load_data
from common import *


def check_hidden_weights(h_w: np.ndarray, x_sample: np.ndarray, t_sample: np.ndarray, network: mlp.ShallowNetwork) \
        -> tuple[float, np.ndarray]:
    """
    Check the effect of back propagation in a given network's hidden weights.
    Wraps a network instance to a function that is accepted by the provided check_gradient function.
    :param h_w: a numpy array containing the current hidden weight matrix
    :param x_sample: a numpy array containing the data points
    :param t_sample: a numpy array containing the labels for the data points
    :param network: the network instance to be wrapped
    :return: a numpy array containing the error and another containing the gradient
    """
    parameters = {"h_w": h_w, "h_b": network.h_b, "o_w": network.o_w, "o_b": network.o_b}
    results = network._back_propagation(x_sample, t_sample, parameters)
    gradient = results[0]
    error = results[4]
    return error, gradient


def check_output_weights(o_w: np.ndarray, x_sample: np.ndarray, t_sample: np.ndarray, network: mlp.ShallowNetwork) \
        -> tuple[float, np.ndarray]:
    """
    Check the effect of back propagation in a given network's output weights.
    Wraps a network instance to a function that is accepted by the provided check_gradient function.
    :param o_w: a numpy array containing the current weight
    :param x_sample: a numpy array containing the data points
    :param t_sample: a numpy array containing the labels for the data points
    :param network: the network instance to be wrapped
    :return: a numpy array containing the error and another containing the gradient
    """
    parameters = {"h_w": network.h_w, "h_b": network.h_b, "o_w": o_w, "o_b": network.o_b}
    results = network._back_propagation(x_sample, t_sample, parameters)
    gradient = results[2].reshape(-1, 1)
    error = results[4]
    return error, gradient


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


def test_grad():
    data = load_data(42)
    h_w_init = np.zeros((784, 25))
    o_w_init = np.zeros((25, 1))

    network = mlp.ShallowNetwork(input_size=784, hidden_size=25, output_size=1, eta=0.3, patience=5,
                                 tolerance=1e-3, activation_func=sigmoid, activation_func_prime=sigmoid_prime,
                                 cost_func=binary_x_entropy, cost_func_prime=binary_x_entropy_prime)
    network.train(data.x_train, data.y_train, data.x_valid, data.y_valid)

    def curried_hidden_wrap(w, x_sample, t_sample): return check_hidden_weights(w, x_sample, t_sample, network=network)
    def curried_output_wrap(w, x_sample, t_sample): return check_output_weights(w, x_sample, t_sample, network=network)

    grad_ew, numerical_grad = check_gradient(h_w_init, data.x_train, data.y_train, curried_hidden_wrap)
    diff = np.max(np.abs(grad_ew - numerical_grad))
    print("The difference estimate for the gradient of the hidden weights is : ", diff)

    grad_ew, numerical_grad = check_gradient(o_w_init, data.x_train, data.y_train, curried_output_wrap)
    diff = np.max(np.abs(grad_ew - numerical_grad))
    print("The difference estimate for the gradient of the output weights is : ", diff)


test_grad()
