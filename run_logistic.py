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
    # transform answers back to array with values of 5 and 6 for comparison
    true_predictions = np.count_nonzero(np.where(predicted_labels == 0, 5, 6) == actual_labels.reshape((-1, 1)))
    return true_predictions / actual_labels.shape[0]


data = load_data()
classifier = LogisticRegClassifier(iters=2000, alpha=0.2, lamda=0, print_history=True)

# transform data into binary array for training
y_train = np.where(data.y_train == 5, 0, 1)
train_cost_history = classifier.train(data.x_train, y_train)

y_test = np.where(data.y_test == 5, 0, 1)
test_cost_history = classifier.test(data.x_test, y_test)

# training results
labels, _ = classifier.predict(data.x_train)
print("Training accuracy: ", get_accuracy(labels, data.y_train))

# testing results
labels, _ = classifier.predict(data.x_test)
print("Testing accuracy: ", get_accuracy(labels, data.y_test))

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
print("Train/Test error figured saved successfully")

