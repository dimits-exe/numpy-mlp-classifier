from mlp import ShallowNetwork
from load_mnist import load_data
from common import get_accuracy, sigmoid, sigmoid_prime, binary_x_entropy, binary_x_entropy_prime

import numpy as np
import matplotlib.pyplot as plt
import os

data = load_data()
classifier = ShallowNetwork(input_size=784, hidden_size=25, output_size=1, eta=0.2, patience=5, tolerance=1e-3,
                            activation_func=sigmoid, activation_func_prime=sigmoid_prime, cost_func=binary_x_entropy,
                            cost_func_prime=binary_x_entropy_prime)

epochs, last_error, train_cost_history = classifier.gradient_descent(data.x_train, data.y_train)

# Train error plot
plt.plot(np.arange(0, epochs, 1), train_cost_history, color="red")
plt.ylabel("Cost")
plt.xlabel("Number of iterations")
plt.title("Mean Binary Cross Entropy Loss")

# Training results
labels = classifier.predict(data.x_train)
print("Training accuracy: ", round(get_accuracy(labels, data.y_train), 3))

# Testing results
labels = classifier.predict(data.x_test)
print("Testing accuracy: ", round(get_accuracy(labels, data.y_test), 3))

plt.show()
plt.savefig(os.path.join("images", "mlp_error.png"))
print("Train error figure saved successfully")

