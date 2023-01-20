import mlp
import numpy as np


class StochasticNetwork(mlp.ShallowNetwork):

    def __init__(self, b: int, **kwargs):
        super(StochasticNetwork, self).__init__(**kwargs)
        self.batch_size = b

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
        # avoid altering original data and labels
        train_data_copy = train_data.copy()
        train_labels_copy = train_labels.copy()

        epoch: int = 0
        least_error: float = np.inf
        epochs_since_improvement: int = 0
        best_model_params = {}
        error_history = []

        h_w = self.h_w.copy()
        o_w = self.o_w.copy()
        h_b = self.h_b.copy()
        o_b = self.o_b.copy()

        while epochs_since_improvement <= self.patience:
            # create batches
            shuffled_data, shuffled_labels = unison_shuffled_copies(train_data_copy, train_labels_copy)
            batch_data = shuffled_data[:self.batch_size]
            batch_labels = shuffled_labels[:self.batch_size]
            reg = train_data.shape[0] / self.batch_size  # gradient regularization

            parameters = {"h_w": h_w, "o_w": o_w, "h_b": h_b, "o_b": o_b}
            dw1, dw2, db1, db2, cost = self._back_propagation(batch_data, batch_labels, parameters)
            error_history.append(cost)

            h_w -= self.eta * reg * dw1
            o_w -= self.eta * reg * dw2
            h_b -= self.eta * reg * db1
            o_b -= self.eta * reg * db2

            # early stopping
            val_loss: float = self._forward_pass(parameters, val_data, val_labels)[4]
            if val_loss + self.tolerance < least_error:
                # print(f"Iteration {epoch} improvement from {least_error} to {error}")
                least_error = val_loss
                epochs_since_improvement = 0
                best_model_params: dict[str, np.ndarray] = parameters
            else:
                epochs_since_improvement += 1
                # print(f"Iteration {epoch} NO improvement from {least_error} to {error}, "
                # f"increasing to {epochs_since_improvement}")

            epoch += 1

        return best_model_params, epoch, least_error, error_history


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
