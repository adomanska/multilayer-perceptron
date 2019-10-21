from neural_network.neural_network import NeuralNetwork
from activation_functions.sigmoid import Sigmoid
from activation_functions.re_lu import ReLU
import numpy as np
from data_transformations import create_test_data, create_train_data, ProblemType

nn = NeuralNetwork()
nn.create_and_add_layer(2, 10, Sigmoid(), False)
nn.create_and_add_layer(10, 15, Sigmoid(), False)
nn.create_and_add_layer(15, 2, Sigmoid(), True)

train_data = create_train_data(ProblemType.Classification, "./data/classification/train/data.simple.train.1000.csv", ["x", "y"], ["cls"])
test_data = create_test_data(ProblemType.Classification, "./data/classification/test/data.simple.test.1000.csv", ["x", "y"], ["cls"])

nn.train(train_data, 10, 100, 0.5, test_data)
