from sgd import StochasticNetwork
from load_mnist import load_data
from common import get_accuracy, sigmoid, sigmoid_prime, binary_x_entropy, binary_x_entropy_prime

INPUT_SIZE = 784
OUTPUT_SIZE = 1
PATIENCE = 5
TOLERANCE = 1e-6

classifier = StochasticNetwork(input_size=INPUT_SIZE, hidden_size=25, output_size=OUTPUT_SIZE, eta=0.1,
                               patience=PATIENCE, tolerance=TOLERANCE,
                               activation_func=sigmoid, activation_func_prime=sigmoid_prime, cost_func=binary_x_entropy,
                               cost_func_prime=binary_x_entropy_prime, b=32)

data = load_data(42)
classifier.train(data.x_train, data.y_train)
labels = classifier.predict(data.x_test)
print("Testing accuracy: ", round(get_accuracy(labels, data.y_test), 3))
