import numpy as np
from logistic_regression import LogisticRegClassifier
from load_mnist import load_data

# debug
# np.seterr(all='raise')

data = load_data()
classifier = LogisticRegClassifier(iters=2000, alpha=0.01, lamda=0, print_history=True)

y = np.where(data.y_train == 5, 0, 1)
cost_history = classifier.train(data.x_train, y)
# print(cost_history)

# training results
labels, _ = classifier.predict(data.x_train)
true_predictions = np.count_nonzero(np.where(labels == 0, 5, 6) == data.y_train.reshape((-1, 1)))
accuracy = true_predictions / data.x_train.shape[0]
print("Training accuracy: ", accuracy)

# testing results
labels, _ = classifier.predict(data.x_test)
true_predictions = np.count_nonzero(np.where(labels == 0, 5, 6) == data.y_test.reshape((-1, 1)))
accuracy = true_predictions / data.x_test.shape[0]
print("Testing accuracy: ", accuracy)
