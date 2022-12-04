from unittest import TestCase
from logistic_regression import LogisticRegClassifier
import load_mnist
import numpy as np


class LogisticRegClassifierTest(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_mnist.load_data()
        cls.classifier = LogisticRegClassifier(iters=500, alpha=0.01, lamda=0.1, print_history=False)

    def test_train(self):
        cost_history = self.classifier.train(self.data.x_train, self.data.y_train)

        assert not array_is_same(np.array(cost_history))
        assert not array_is_same(self.classifier.weights)

    def test_test(self):
        self.classifier.train(self.data.x_train, self.data.y_train)
        cost_history = self.classifier.train(self.data.x_test, self.data.y_test)
        assert not array_is_same(np.array(cost_history))
        assert not array_is_same(self.classifier.weights)


def array_is_same(array: np.ndarray) -> bool:
    return np.all(np.isclose(array, array[0]))
