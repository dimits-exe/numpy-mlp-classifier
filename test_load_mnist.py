from unittest import TestCase
import load_mnist
import numpy as np


class LoadTest(TestCase):
    """
    Tests the basic requirements for the shape and contents of our subset of the MNIST dataset.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.mnist_data = load_mnist.load_data()

    def test_image_shape(self):
        assert self.mnist_data.x_train.shape[1] == 784
        assert self.mnist_data.x_test.shape[1] == 784
        assert self.mnist_data.x_valid.shape[1] == 784

    def test_data_label_eq(self):
        assert self.mnist_data.x_train.shape == self.mnist_data.y_train.shape
        assert self.mnist_data.x_test.shape == self.mnist_data.y_test.shape
        assert self.mnist_data.x_valid.shape == self.mnist_data.y_valid.shape

    def test_data_range(self):
        assert np.all(self.mnist_data.x_train >= 0 & self.mnist_data.x_train <= 1)
        assert np.all(self.mnist_data.x_test >= 0 & self.mnist_data.y_test <= 1)
        assert np.all(self.mnist_data.x_valid >= 0 & self.mnist_data.x_valid <= 1)

    def test_validation_size(self):
        assert np.isclose(len(self.mnist_data.x_train), len(self.mnist_data.x_valid) * 5, rtol=2)


