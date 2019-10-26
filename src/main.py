from neural_network.regression_neural_network import RegressionNeuralNetwork
from neural_network.classification_neural_network import ClassificationNeuralNetwork
from activation_functions.sigmoid import Sigmoid
from activation_functions.re_lu import ReLU
from activation_functions.linear import Linear
from activation_functions.tan_h import TanH
from cost_functions.quadratic_cost import QuadraticCost
import numpy as np
from data_transformations import create_test_data, create_train_data, ProblemType
import matplotlib.pyplot as plt

# Classification
print("Classification")
nn = ClassificationNeuralNetwork()
nn.create_and_add_layer(2, 10, Sigmoid(), False)
nn.create_and_add_layer(10, 15, Sigmoid(), False)
nn.create_and_add_layer(15, 2, Sigmoid(), True, QuadraticCost())

train_data = create_train_data(ProblemType.Classification, "./data/classification/train/data.simple.train.1000.csv", ["x", "y"], ["cls"])
test_data = create_test_data(ProblemType.Classification, "./data/classification/test/data.simple.test.1000.csv", ["x", "y"], ["cls"])
nn.train(train_data, 10, 100, 0.5, test_data)

# Regression
# print("Regression")
# nn = RegressionNeuralNetwork()
# nn.create_and_add_layer(1, 9, Sigmoid(), False)
# nn.create_and_add_layer(9, 1, Linear(), True)

# train_data = create_train_data(ProblemType.Regression, "./data/regression/train/data.activation.train.1000.csv", ["x"], ["y"])
# test_data = create_test_data(ProblemType.Regression, "./data/regression/test/data.activation.test.1000.csv", ["x"], ["y"])

# sorted_data = sorted(test_data, key=lambda x: x[0][0])
# xs = np.array([x[0] for x, y in sorted_data])
# ys = np.array([y[0] for x, y in sorted_data])

# nn.train(train_data, 10, 1000, 0.006, test_data)
# yp = [nn._feed_forward(np.array([x]))[0] for x in xs]

# plt.plot(xs, ys)
# plt.plot(xs, yp)
# plt.show()
