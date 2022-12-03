from unittest import TestCase
from logistic_regression import LogisticRegClassifier
import load_mnist


class LogisticRegClassifierTest(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_mnist.load_data()
        cls.classifier = LogisticRegClassifier(iters=500, alpha=0.01, lamda=0.1, print_history=False)

    def test_train(self):
        self.classifier.train(self.data.x_train, self.data.y_train)

    def test_test(self):
        self.classifier.train(self.data.x_test, self.data.y_test)

