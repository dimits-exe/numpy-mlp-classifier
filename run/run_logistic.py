from models.logistic_regression import LogisticRegClassifier
from lib.load_mnist import load_data
from lib.common import get_accuracy

import time
import numpy as np
import matplotlib.pyplot as plt
import os

start_time = time.process_time()
ITERATIONS = 250
LEARNING_RATE = 0.2

# Extreme lambdas cause arithmetic overflow.
# As its normal that models with extreme regularization don't learn,
# we prevent numpy warnings from being spammed in the console.
np.seterr(all="ignore")

print("Loading data...")
data = load_data()

print("Training classifier for the train-loss figure...")
classifier = LogisticRegClassifier(iters=ITERATIONS*2, alpha=LEARNING_RATE, lamda=0, print_history=True)

train_cost_history = classifier.train(data.x_train, data.y_train)
print("Testing trained classifier for the test-loss figure...")
test_cost_history = classifier.test(data.x_test, data.y_test)

# training results
labels, _ = classifier.predict(data.x_train)
print("Training accuracy: ", get_accuracy(labels, data.y_train))

# testing results
labels, _ = classifier.predict(data.x_test)
print("Testing accuracy: ", round(get_accuracy(labels, data.y_test), 3))

# Save train/test accuracy plot
print("Producing train/test loss figure")
fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained")
fig.suptitle("Logistic Regression Classifier Error")

ax1.plot(np.arange(0, classifier.iters + 1, 1), train_cost_history, color="red")
ax1.set_ylabel("Cost")
ax1.set_xlabel("Number of iterations")
ax1.set_title("Training")

ax2.plot(np.arange(0, classifier.iters + 1, 1), test_cost_history, color="red")
ax2.set_ylabel("Cost")
ax2.set_xlabel("Number of iterations")
ax2.set_title("Testing")

path = os.path.join("../images", "mlp_error.png")
plt.savefig(path)
print("Train/Test loss plot saved in ", path)

# Search for optimal lambda parameter
lamda_search_count = 100
val_loss = np.full(lamda_search_count, fill_value=-np.inf)
lamda_values = np.logspace(start=-4, stop=1, num=lamda_search_count)

print("Starting parameter search for optimal lambda")
for i in range(lamda_search_count):
    if i % 10 == 0:
        print(str(i) + "% complete...")

    classifier = LogisticRegClassifier(iters=ITERATIONS, alpha=LEARNING_RATE, lamda=lamda_values[i], print_history=False)
    classifier.train(data.x_train, data.y_train)

    val_loss[i] = classifier.test(data.x_valid, data.y_valid).mean()

    predicted, _ = classifier.predict(data.x_valid)
    accuracy = get_accuracy(predicted, data.y_valid)

best_index = np.nanargmax(val_loss)  # find the largest loss, since it's negative due to gradient descent
best_lambda = lamda_values[best_index]
print(f"Best lambda value {best_lambda} with validation accuracy={val_loss[best_index]}")

# Test on optimal lambda
classifier = LogisticRegClassifier(iters=ITERATIONS, alpha=LEARNING_RATE, lamda=best_lambda, print_history=False)
classifier.train(data.x_train, data.y_train)
predicted, _ = classifier.predict(data.x_test)
best_lambda_test_acc = get_accuracy(predicted, data.y_test)
print(f"Test accuracy for optimal lambda={best_lambda}: {best_lambda_test_acc}")

# Save lambda plot
plt.figure()
plt.title("Logistic Regression Classifier Validation Loss")
plt.plot(lamda_values, val_loss, color="red", marker="D")
plt.xlabel("Lambda value")
plt.ylabel("Validation Loss")

plt.savefig(os.path.join("../images", "logistic_lambda_accuracy.png"))
print("Lambda-loss plot saved successfully")

end_time = time.process_time()
print(f"Tasks finished after {end_time - start_time} seconds")
