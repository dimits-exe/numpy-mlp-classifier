import tensorflow.keras.datasets.mnist as mnist
import numpy as np


class Data:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                 x_valid: np.ndarray, y_valid: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_valid = x_valid
        self.y_valid = y_valid

    def __str__(self):
        return f"Train: {self.x_train.shape} with {self.y_train.shape} labels\n" +\
                f"Test: {self.x_test.shape} with {self.y_test.shape} labels\n" +\
                f"Validation: {self.x_valid.shape} with {self.y_valid.shape} labels\n"


def load_data(seed: int = -1) -> Data:
    """
    Load the MNIST dataset, filter out data that aren't "5" or "6", rescale their values and returns the train,
    test and validation data for the MLP as 784 pixel vectors.
    :return a Data object holding the train test and validation data
    """
    if seed != -1:
        np.random.seed(seed)

    data = mnist.load_data()

    # randomly shuffle all arrays
    for example_category in data:
        _reshuffle_data(example_category[0], example_category[1])

    x_train = _rescale_data(_array_to_vector(data[0][0]))
    y_train = data[0][1]
    x_test = _rescale_data(_array_to_vector(data[1][0]))
    y_test = data[1][1]

    x_train, y_train = _filter_data(x_train, y_train)
    x_test, y_test = _filter_data(x_test, y_test)

    x_train, x_valid, = _split_to_validation(x_train)
    y_train, y_valid = _split_to_validation(y_train)

    y_train = _get_binary_labels(y_train)
    y_test = _get_binary_labels(y_test)
    y_valid = _get_binary_labels(y_valid)

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    y_valid = y_valid.reshape((-1, 1))

    return Data(x_train, y_train, x_test, y_test, x_valid, y_valid)


def _reshuffle_data(data: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshuffle the data and its labels while preserving the relations between them.
    :param data: a numpy array containing the data
    :param labels: a numpy array containing the data's labels
    :return: the same arrays, reshuffled
    """
    # this does create and return copies, but shouldn't be a problem
    assert len(data) == len(labels)
    indexes = np.random.permutation(len(data))
    return data[indexes], labels[indexes]


def _get_binary_labels(labels: np.ndarray) -> np.ndarray:
    """
    Get a binary numpy array where the ith element is 0 if it was "5" and 1 otherwise
    :param labels: a numpy array containing the data's labels
    :return: a binary numpy array
    """
    return np.where(labels == 5, 0, 1)


def _filter_data(data: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Delete records from the data and label vectors that are not "5" or "6"
    :param data: a vector containing the data
    :param labels: a vector containing the data's labels
    :return: a tuple containing the new data its labels
    """
    condition = (labels == 5) | (labels == 6)
    data = data[condition]
    labels = labels[condition]
    return data, labels


def _split_to_validation(array: np.ndarray, split_perc: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Split data into training and validation data.
    :param array: the data
    :param split_perc: the percent of the data that will be allocated for the validation set
    :return: a tuple containing the new training and validation data
    """
    validation_value_count = int(array.shape[0] * split_perc)
    return array[validation_value_count:], array[:validation_value_count]


def _array_to_vector(array: np.ndarray):
    """
    Converts an array holding k mxn sized matrices into an array containing k vectors.
    :param array: the array holding the matrices
    :return: an array holding the 1D vectors
    """
    return np.reshape(array, (array.shape[0], array.shape[1] * array.shape[2]))


def _rescale_data(array: np.ndarray):
    """
    Rescales all elements of the array, so they lie in the [0,1] range
    :param array: the data
    :return: the rescaled data
    """
    return array / 255
