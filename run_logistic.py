from logistic_regression import LogisticRegClassifier
from load_mnist import load_data
from common import get_accuracy

import time
import numpy as np
import matplotlib.pyplot as plt
import os

start_time = time.process_time()
iterations = 500
learning_rate = 0.2

data = load_data()
classifier = LogisticRegClassifier(iters=iterations, alpha=learning_rate, lamda=0, print_history=True)

train_cost_history = classifier.train(data.x_train, data.y_train)
test_cost_history = classifier.test(data.x_test, data.y_test)

# training results
labels, _ = classifier.predict(data.x_train)
print("Training accuracy: ", get_accuracy(labels, data.y_train))

# testing results
labels, _ = classifier.predict(data.x_test)
print("Testing accuracy: ", round(get_accuracy(labels, data.y_test), 3))

# Save train/test accuracy plot

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

plt.savefig(os.path.join("images", "logistic_error.png"))
print("Train/Test error figure saved successfully")

# Search for optimal lambda parameter

val_loss = np.zeros(100)
lamda_values = np.logspace(start=-4, stop=1, num=100)

print("Starting parameter search for optimal lambda")
for i, lamda in enumerate(lamda_values):
    if i % 10 == 0:
        print(str(i) + "% complete...")

    classifier = LogisticRegClassifier(iters=iterations, alpha=learning_rate, lamda=lamda, print_history=False)
    classifier.train(data.x_train, data.y_train)
    val_loss[i] = classifier.test(data.x_valid, data.y_valid).mean()

best_index = val_loss.argmin()
best_lambda = lamda_values[best_index]
print(f"Best lambda value {best_lambda} with validation accuracy={val_loss[best_index]}")

# Test on optimal lambda
classifier = LogisticRegClassifier(iters=iterations, alpha=learning_rate, lamda=best_lambda, print_history=False)
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

plt.savefig(os.path.join("images", "logistic_lambda_accuracy.png"))
print("Lambda-loss plot saved successfully")

end_time = time.process_time()
print(f"Tasks finished after {end_time - start_time} seconds")
