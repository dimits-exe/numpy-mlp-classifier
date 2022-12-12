import numpy as np
import matplotlib.pyplot as plt
import os
from logistic_regression import LogisticRegClassifier
from load_mnist import load_data


def get_accuracy(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> float:
    """
    Get the accuracy of the model based on its predicted and actual labels of its data.
    :param predicted_labels: the labels which the model predicted
    :param actual_labels: the actual labels of the data
    :return: a number between 0 and 1 representing the accuracy of the model
    """
    true_predictions = np.count_nonzero(np.where(predicted_labels == 0, 0, 1) == actual_labels.reshape((-1, 1)))
    return true_predictions / actual_labels.shape[0]

# Train and test classifier


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

accuracy = np.zeros(100)
lamda_values = np.logspace(start=-4, stop=1, num=100)

print("Starting parameter search for optimal lambda")
for i, lamda in enumerate(lamda_values):
    if i % 10 == 0:
        print(i, "% complete...")

    classifier = LogisticRegClassifier(iters=iterations, alpha=learning_rate, lamda=lamda, print_history=False)
    classifier.train(data.x_train, data.y_train)
    predicted, _ = classifier.predict(data.x_valid)
    accuracy[i] = get_accuracy(predicted, data.y_valid)

best_index = accuracy.argmax()
best_lambda = lamda_values[best_index]
print(f"Best lambda value {best_lambda} with test accuracy={accuracy[best_index]}")

# Test on optimal lambda

classifier = LogisticRegClassifier(iters=iterations, alpha=learning_rate, lamda=best_lambda, print_history=False)
classifier.train(data.x_train, data.y_train)
predicted, _ = classifier.predict(data.x_test)
best_lambda_test_acc = get_accuracy(predicted, data.y_test)
print(f"Test accuracy for optimal lambda={best_lambda}: {best_lambda_test_acc}")

# Save lambda plot

plt.figure()
plt.title("Logistic Regression Classifier Test Accuracy")
plt.plot(lamda_values, accuracy * 100, color="red", marker="D")
plt.xlabel("Lambda value")
plt.ylabel("Validation test accuracy (%)")

plt.savefig(os.path.join("images", "logistic_lambda_accuracy.png"))
print("Lambda-accuracy plot saved successfully")
