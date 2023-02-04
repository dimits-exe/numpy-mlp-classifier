# numpy-mlp-classifier

A project implementing a Logistic Regression classifier, a shallow MLP neural network and a batch stochastic gradient descent variation
using numpy. 


## Installation

You need to download the `numpy` and `matplotlib` libraries, as well as `tensorflow` if you wish to use the preproccessed datasets 
included in this project. To do so run the following commands in your terminal (pip must be installed):
- `pip install numpy`
- `pip install matplotlib`
- `pip install tensorflow`

Then download the project.


## Usage

An example of training and using a batch stochastic MLP model is provided below:

```py
from models.mlp import ShallowNetwork
from lib.load_mnist import load_data
from lib.common import get_accuracy, sigmoid, sigmoid_prime, binary_x_entropy, binary_x_entropy_prime

import numpy as np
import matplotlib.pyplot as plt

INPUT_SIZE = 784
OUTPUT_SIZE = 1
PATIENCE = 5
TOLERANCE = 1e-3

print("Loading data...")
data = load_data()

print("Training classifier...")
classifier = ShallowNetwork(input_size=INPUT_SIZE, hidden_size=25, output_size=OUTPUT_SIZE, eta=0.2,
                            patience=PATIENCE, tolerance=TOLERANCE,
                            activation_func=sigmoid, activation_func_prime=sigmoid_prime, cost_func=binary_x_entropy,
                            cost_func_prime=binary_x_entropy_prime)
epochs, val_error, train_cost_history = classifier.train(data.x_train, data.y_train, data.x_valid, data.y_valid)

# Training results
print("Mean validation loss: ", val_error)
train_labels, _ = classifier.predict(data.x_train)
print("Training accuracy: ", round(get_accuracy(train_labels, data.y_train), 3))

# Testing results
test_labels, test_error = classifier.predict(data.x_test, data.y_test)
print("Mean testing loss: ", test_error)
print("Testing accuracy: ", round(get_accuracy(test_labels, data.y_test), 3))
```

A preproccessed dataset is available by using the `load_mnist.py` file in the `lib` module.

You can use each of the models by importing them from the `models` module. API documentation is provided in the form of docstrings in the
source files. 

The `run` module contains executable code that displays various use cases such as using the classifiers for hyper-parameter grid search.

## Documentation

A high-level overview of the project is provided in the `documentation.pdf` file, including notes, observations, graphs, and performance
characteristics.
