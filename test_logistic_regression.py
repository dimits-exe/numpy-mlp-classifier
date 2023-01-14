from logistic_regression import LogisticRegClassifier
from common import get_accuracy
import load_mnist

from unittest import TestCase
import numpy as np


class LogisticRegClassifierTest(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_mnist.load_data()

    def setUp(self) -> None:
        self.classifier = LogisticRegClassifier(iters=500, alpha=0.01, lamda=0.1, print_history=False)

    def test_train(self):
        cost_history = self.classifier.train(self.data.x_train, self.data.y_train)

        assert not array_is_same(np.array(cost_history))
        assert not array_is_same(self.classifier.weights)

    def test_test(self):
        self.classifier.train(self.data.x_train, self.data.y_train)
        cost_history = self.classifier.train(self.data.x_test, self.data.y_test)
        assert not array_is_same(np.array(cost_history))
        assert not array_is_same(self.classifier.weights)

    def test_accuracy(self):
        self.classifier.train(self.data.x_train, self.data.y_train)
        predicted_train, _ = self.classifier.predict(self.data.x_train)
        predicted_test, _ = self.classifier.predict(self.data.x_test)

        accuracy_train = get_accuracy(predicted_train, self.data.y_train)
        accuracy_test = get_accuracy(predicted_test, self.data.y_test)

        assert accuracy_train > 0.75
        assert accuracy_test > 0.75


def array_is_same(array: np.ndarray) -> bool:
    return np.all(np.isclose(array, array[0]))
