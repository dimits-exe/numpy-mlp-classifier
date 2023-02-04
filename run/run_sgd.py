from models.sgd import StochasticNetwork
from lib.load_mnist import load_data
from lib.common import get_accuracy, sigmoid, sigmoid_prime, binary_x_entropy, binary_x_entropy_prime

import time
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_SIZE = 784
OUTPUT_SIZE = 1
PATIENCE = 5
TOLERANCE = 1e-3
HIDDEN_SIZE = 25
LEARNING_RATE = 0.2

start_time = time.process_time()
np.seterr(all="ignore")  # not debug
print("Loading data...")
data = load_data(130)

print("Training classifier for train loss figure...")
classifier = StochasticNetwork(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE,
                               eta=LEARNING_RATE, patience=PATIENCE, tolerance=TOLERANCE,
                               activation_func=sigmoid, activation_func_prime=sigmoid_prime, cost_func=binary_x_entropy,
                               cost_func_prime=binary_x_entropy_prime, b=256)
epochs, val_loss, train_cost_history = classifier.train(data.x_train, data.y_train, data.x_valid, data.y_valid)

# Training results
train_labels, _ = classifier.predict(data.x_train)
print("Mean validation loss: ", val_loss)
print("Training accuracy: ", round(get_accuracy(train_labels, data.y_train), 3))

# Testing results
test_labels, test_error = classifier.predict(data.x_test, data.y_test)
print("Test error: ", test_error)
print("Testing accuracy: ", round(get_accuracy(test_labels, data.y_test), 3))

# Train error plot
print("Creating train loss plot...")
plt.plot(np.arange(0, epochs, 1), train_cost_history, color="red")
plt.ylabel("Cost")
plt.xlabel("Number of iterations")
plt.title("Mean Binary Cross Entropy Loss")

path = os.path.join("../images", "sgd_error.png")
plt.savefig(path)
print("Train-Loss plot saved in ", path)

# Parameter search
b_search_count = 8
val_loss = np.zeros(b_search_count)
epochs_needed = np.zeros(b_search_count).astype(int)
b_values = np.logspace(start=1, stop=8, num=8, base=2).astype(int)

print("Starting parameter search for optimal beta value.")
for i in range(b_search_count):
    b = b_values[i]

    classifier = StochasticNetwork(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE,
                                   eta=0.3, patience=PATIENCE, tolerance=TOLERANCE,
                                   activation_func=sigmoid, activation_func_prime=sigmoid_prime,
                                   cost_func=binary_x_entropy,
                                   cost_func_prime=binary_x_entropy_prime, b=b)
    epochs, _, _ = classifier.train(data.x_train, data.y_train, data.x_valid, data.y_valid)
    epochs_needed[i] = epochs
    labels, val_loss[i] = classifier.predict(data.x_valid, data.y_valid)

best_index = np.unravel_index(val_loss.argmin(), val_loss.shape)
best_b = b_values[best_index[0]]

print(f"Best hyper-parameter b={best_b},"
      f"trained on {epochs_needed[best_index]} epochs and "
      f"with validation loss={val_loss[best_index]}")


# Test optimal network
print("Training classifier with optimal hyper-parameters...")
classifier = StochasticNetwork(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE,
                               eta=0.3, patience=PATIENCE, tolerance=TOLERANCE, activation_func=sigmoid,
                               activation_func_prime=sigmoid_prime, cost_func=binary_x_entropy,
                               cost_func_prime=binary_x_entropy_prime, b=best_b)

classifier.train(data.x_train, data.y_train, data.x_valid, data.y_valid)
optimal_test_labels, optimal_test_loss = classifier.predict(data.x_test, data.y_test)
print("Test loss for optimal hyper-parameters: ", optimal_test_loss)
print("Test accuracy for optimal hyper-parameters: ", round(get_accuracy(optimal_test_labels, data.y_test), 3))

# Parameter search for optimal eta and m values
# Same code as run_mlp.py
eta_search_count = 10
m_search_count = 10
val_loss = np.zeros((eta_search_count, m_search_count))
epochs_needed = np.zeros((eta_search_count, m_search_count)).astype(int)
eta_values = np.logspace(start=0, stop=-5, num=eta_search_count, base=10) / 2
m_values = np.logspace(start=1, stop=10, num=m_search_count, base=2).astype(int)

print("Starting parameter search for optimal eta and M values.")
for i in range(m_search_count):
    print(str(i * 10) + "% complete...")
    for j in range(eta_search_count):
        m = m_values[i]
        eta = eta_values[j]

        classifier = StochasticNetwork(input_size=INPUT_SIZE, hidden_size=m, output_size=OUTPUT_SIZE, eta=eta,
                                       patience=PATIENCE, tolerance=TOLERANCE, activation_func=sigmoid,
                                       activation_func_prime=sigmoid_prime, cost_func=binary_x_entropy,
                                       cost_func_prime=binary_x_entropy_prime, b=best_b)
        epochs, _, _ = classifier.train(data.x_train, data.y_train, data.x_valid, data.y_valid)
        epochs_needed[i][j] = epochs
        val_loss[i][j] = classifier.predict(data.x_valid, data.y_valid)[1]

best_index = np.unravel_index(np.nanargmin(val_loss), val_loss.shape)
best_m = m_values[best_index[0]]
best_eta = eta_values[best_index[1]]

print(f"Best hyper-parameters: eta={best_eta}, m={best_m} "
      f"trained on {epochs_needed[best_index[0], best_index[1]]} epochs and "
      f"with validation loss={val_loss[best_index[0], best_index[1]]}")

# Test optimal network
print("Training classifier with optimal hyper-parameters...")
classifier = StochasticNetwork(input_size=INPUT_SIZE, hidden_size=best_m, output_size=OUTPUT_SIZE, eta=best_eta,
                               patience=PATIENCE, tolerance=TOLERANCE, activation_func=sigmoid,
                               activation_func_prime=sigmoid_prime, cost_func=binary_x_entropy,
                               cost_func_prime=binary_x_entropy_prime, b=best_b)
classifier.train(data.x_train, data.y_train, data.x_valid, data.y_valid)
optimal_test_labels, optimal_test_loss = classifier.predict(data.x_test, data.y_test)
print("Test loss for optimal hyper-parameters: ", optimal_test_loss)
print("Test accuracy for optimal hyper-parameters: ", round(get_accuracy(optimal_test_labels, data.y_test), 3))

end_time = time.process_time()
print(f"Tasks finished after {end_time - start_time} seconds")
