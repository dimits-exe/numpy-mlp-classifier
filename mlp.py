import numpy as np
from load_mnist import load_data


class ShallowNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, eta: float, epochs: int):
        self.input_size = input_size
        self.eta = eta
        self.epochs = epochs

        # don't set weights and biases for the input
        self.h_b = np.zeros((1, hidden_size))
        self.o_b = np.zeros((1, output_size))

        self.h_w = np.random.random((input_size, hidden_size))
        self.o_w = np.random.random((hidden_size, output_size))

    def gradient_descent(self, train_data: np.ndarray, train_labels: np.ndarray) -> None:
        hidden_weight_shape = self.h_w.shape
        output_weight_shape = self.o_w.shape
        hidden_bias_shape = self.h_b.shape
        output_bias_shape = self.o_b.shape

        for epoch in range(self.epochs + 1):
            dw1, dw2, db1, db2, cost = self.back_propagation(train_data, train_labels)

            if epoch % 50 == 0:
                print(f"Iteration {epoch} Error: {cost}")

            self.h_w -= self.eta * dw1
            self.o_w -= self.eta * dw2
            self.h_b -= self.eta * db1
            self.o_b -= self.eta * db2

            assert self.h_w.shape == hidden_weight_shape
            assert self.o_w.shape == output_weight_shape
            assert self.h_b.shape == hidden_bias_shape
            assert self.o_b.shape == output_bias_shape

    def back_propagation(self, x: np.ndarray, y: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:

        # Forward pass
        z1 = x.dot(self.h_w) + self.h_b
        a1 = sigmoid(z1)
        z2 = a1.dot(self.o_w) + self.o_b
        a2 = sigmoid(z2)

        # Backward pass
        m = x.shape[1]

        # output layer activation derivative
        dy_hat = a2 - y.reshape((-1, 1))
        dz2 = dy_hat * sig_prime(z2)
        dw2: np.ndarray = (1/m) * a1.T.dot(dz2)
        db2: np.ndarray = (1/m) * np.sum(dz2, axis=0)

        # hidden layer activation derivative
        da1 = dz2.dot(self.o_w.T)
        dz1 = da1 * sig_prime(z1)
        dw1: np.ndarray = (1/m) * x.T.dot(dz1)
        db1: np.ndarray = (1/m) * np.sum(dz1, axis=0)

        return dw1, dw2, db1, db2, dz2[0]

    def output(self, x: np.ndarray) -> np.ndarray:
        x = sigmoid(x.dot(self.h_w) + self.h_b)
        x = sigmoid(x.dot(self.o_w) + self.o_b)
        return x

    def predict(self, test_data):
        return np.where(self.output(test_data) < 0.5, 0, 1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sig_prime(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def cost_derivative(output, y):
    return output - y


def get_accuracy(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> float:
    """
    Get the accuracy of the model based on its predicted and actual labels of its data.
    :param predicted_labels: the labels which the model predicted
    :param actual_labels: the actual labels of the data
    :return: a number between 0 and 1 representing the accuracy of the model
    """
    true_predictions = np.count_nonzero(np.where(predicted_labels == 0, 0, 1) == actual_labels.reshape((-1, 1)))
    return true_predictions / actual_labels.shape[0]


def main():
    np.seterr(all="raise")
    data = load_data(42)

    m = 25
    network = ShallowNetwork(input_size=784, hidden_size=m, output_size=1, eta=0.1, epochs=100)
    network.gradient_descent(data.x_train, data.y_train)

    train_accuracy = get_accuracy(network.predict(data.x_train), data.y_train.reshape(-1, 1))
    print("Training accuracy: ", train_accuracy)

    test_accuracy = get_accuracy(network.predict(data.x_test), data.y_test.reshape(-1, 1))
    print("Testing accuracy: ", test_accuracy)


if __name__ == "__main__":
    main()


    """
    bias_gradients = [np.zeros(b.shape) for b in self.biases]
        weight_gradients = [np.zeros(w.shape) for w in self.weights]

        x = x.T
        # forward pass
        last_activation = x
        activations = [x]
        z_vectors = []

        for bias, weight in zip(self.biases, self.weights):
            zh = weight.T.dot(last_activation) + bias.T
            z_vectors.append(zh)
            h = sigmoid(zh)
            activations.append(h)
            last_activation = h

        # backward pass
        # since this is a 3 layer mlp, we only need to update the output weights
        cost = cost_derivative(activations[-1], y.reshape(1, -1))
        delta = cost * sig_prime(z_vectors[-1])
        z = z_vectors[-2]
        delta = self.weights[-1].dot(delta) * sig_prime(z)
        # TODO: figure out this mess
        bias_gradients[-2] = delta.mean(axis=1).reshape(1, delta.shape[0])
        # what the fuck
        weight_gradients[-2] = activations[-3].dot(delta.T)

        return bias_gradients, weight_gradients, cost[0]"""
