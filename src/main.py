from neural_network.neural_network import NeuralNetwork
from neural_network.classifier import Classifier
from activation_functions.sigmoid import Sigmoid
from activation_functions.re_lu import ReLU
from visualisator import Visualisator
import numpy as np
from data_reader import DataReader

nn = Classifier()
nn.create_and_add_layer(2, 10, Sigmoid(), False)
nn.create_and_add_layer(10, 15, Sigmoid(), False)
nn.create_and_add_layer(15, 2, Sigmoid(), True)

def create_output(i, n):
    output = np.zeros(n)
    output[i - 1] = 1
    return output

train_data_reader = DataReader("./data/classification/train/data.simple.train.1000.csv")
xs = train_data_reader.get_columns(["x", "y"])
ys = train_data_reader.get_columns(["cls"])
train_data = [(x, create_output(int(y), 2)) for x, y in zip(xs, ys)]
test_data_reader = DataReader("./data/classification/test/data.simple.test.1000.csv")
xs = test_data_reader.get_columns(["x", "y"])
ys = test_data_reader.get_columns(["cls"])
test_data = [(x, y[0] - 1) for x, y in zip(xs, ys)]

nn.train(train_data, 10, 100, 0.5, test_data)

Visualisator.draw_classifier_scatter_plot(nn, test_data)