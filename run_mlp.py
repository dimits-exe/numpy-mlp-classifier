from mlp import ShallowNetwork
from load_mnist import load_data
from common import get_accuracy, sigmoid, sigmoid_prime, binary_x_entropy, binary_x_entropy_prime

import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_SIZE = 784
OUTPUT_SIZE = 1
PATIENCE = 5
TOLERANCE = 1e-3

print("Loading data...")
data = load_data()

print("Training classifier for train loss figure...")
classifier = ShallowNetwork(input_size=INPUT_SIZE, hidden_size=25, output_size=OUTPUT_SIZE, eta=0.2,
                            patience=PATIENCE, tolerance=TOLERANCE,
                            activation_func=sigmoid, activation_func_prime=sigmoid_prime, cost_func=binary_x_entropy,
                            cost_func_prime=binary_x_entropy_prime)
epochs, last_error, train_cost_history = classifier.gradient_descent(data.x_train, data.y_train)

# Training results
labels = classifier.predict(data.x_train)
print("Training accuracy: ", round(get_accuracy(labels, data.y_train), 3))

# Testing results
labels = classifier.predict(data.x_test)
print("Testing accuracy: ", round(get_accuracy(labels, data.y_test), 3))

# Train error plot
print("Creating train loss plot...")
plt.plot(np.arange(0, epochs, 1), train_cost_history, color="red")
plt.ylabel("Cost")
plt.xlabel("Number of iterations")
plt.title("Mean Binary Cross Entropy Loss")

plt.savefig(os.path.join("images", "mlp_error.png"))
print("Train-Loss figure saved successfully.")

# Parameter search
eta_search_count = 10
m_search_count = 10

accuracy = np.zeros((eta_search_count, m_search_count))
epochs_needed = np.zeros((eta_search_count, m_search_count)).astype(int)
eta_values = np.logspace(start=-5, stop=-1, num=eta_search_count, base=10)
m_values = np.logspace(start=1, stop=10, num=m_search_count, base=2).astype(int)

print("Starting parameter search for optimal eta and M values.")
for i in range(m_search_count):
    print(str(i * 10) + "% complete...")
    for j in range(eta_search_count):
        m = m_values[i]
        eta = eta_values[j]

        classifier = ShallowNetwork(input_size=INPUT_SIZE, hidden_size=m, output_size=OUTPUT_SIZE, eta=eta,
                                    patience=PATIENCE, tolerance=TOLERANCE, activation_func=sigmoid,
                                    activation_func_prime=sigmoid_prime, cost_func=binary_x_entropy,
                                    cost_func_prime=binary_x_entropy_prime)
        epochs, _, _ = classifier.gradient_descent(data.x_train, data.y_train)
        predicted = classifier.predict(data.x_valid)
        accuracy[i][j] = get_accuracy(predicted, data.y_valid)
        epochs_needed[i][j] = epochs

best_index = np.unravel_index(accuracy.argmax(), accuracy.shape)
best_eta = eta_values[best_index[0]]
best_m = m_values[best_index[1]]
print(f"Best hyper-parameters: eta={best_eta}, m={best_m} "
      f"trained on {epochs_needed[best_index[0], best_index[1]]} epochs and "
      f"with validation accuracy={accuracy[best_index[0], best_index[1]]}")

# Test optimal network
print("Training classifier with optimal hyper-parameters...")
classifier = ShallowNetwork(input_size=INPUT_SIZE, hidden_size=best_m, output_size=OUTPUT_SIZE, eta=best_eta,
                            patience=PATIENCE, tolerance=TOLERANCE, activation_func=sigmoid,
                            activation_func_prime=sigmoid_prime, cost_func=binary_x_entropy,
                            cost_func_prime=binary_x_entropy_prime)

classifier.gradient_descent(data.x_train, data.y_train)
labels = classifier.predict(data.x_test)
print("Test accuracy for optimal hyper-parameters: ", round(get_accuracy(labels, data.y_test), 3))

