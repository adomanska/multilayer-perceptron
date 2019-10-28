from neural_network.regression_neural_network import RegressionNeuralNetwork
from neural_network.classification_neural_network import ClassificationNeuralNetwork
from activation_functions.sigmoid import Sigmoid
from activation_functions.re_lu import ReLU
from activation_functions.linear import Linear
from activation_functions.tan_h import TanH
from visualisator import Visualisator
from cost_functions.quadratic_cost import QuadraticCost
from report.report import Report
from cost_functions.cross_entropy_cost import CrossEntropyCost
import numpy as np
from data_transformations import create_test_data, create_train_data, ProblemType
import matplotlib.pyplot as plt


# Report.run_tests()

Report.test_regression()
# Classification

# print("Classification")
# nn = ClassificationNeuralNetwork()
# nn.create_and_add_hidden_layer(2, 10, Sigmoid())
# nn.create_and_add_hidden_layer(10, 15, Sigmoid())
# nn.create_and_add_output_layer(15, 2, Sigmoid(), QuadraticCost())

# train_data = create_train_data(ProblemType.Classification, "./data/classification/train/data.three_gauss.train.1000.csv", ["x", "y"], ["cls"])
# test_data = create_test_data(ProblemType.Classification, "./data/classification/test/data.three_gauss.test.1000.csv", ["x", "y"], ["cls"])
# nn.train(train_data, 10, 100, 0.5, 0, test_data)

# Visualisator.visualise_classification(nn, test_data)

# Regression
# print("Regression")
# nn = RegressionNeuralNetwork()
# nn.create_and_add_hidden_layer(1, 9, Sigmoid())
# nn.create_and_add_output_layer(9, 1, Linear(), QuadraticCost())


# train_data = create_train_data(ProblemType.Regression, "./data/regression/train/data.activation.train.1000.csv", ["x"], ["y"])
# test_data = create_test_data(ProblemType.Regression, "./data/regression/test/data.activation.test.1000.csv", ["x"], ["y"])

# nn.train(train_data, 10, 100, 0.006, 0, test_data)

# Visualisator.visualise_regression(nn, test_data)