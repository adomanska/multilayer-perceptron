from neural_network.neural_network import NeuralNetwork
from activation_functions.sigmoid import Sigmoid
from activation_functions.re_lu import ReLU
import numpy as np
from data_reader import DataReader

nn = NeuralNetwork()
nn.create_and_add_layer(2, 10, Sigmoid())
nn.create_and_add_layer(10, 15, Sigmoid())
nn.create_and_add_layer(15, 2, Sigmoid(), True)

# data = np.array([
#     ([0.1, 0.1], [1, 0]),
#     ([0.2, 0.2], [0, 1]),
#     ([0.3, 0.3], [1, 0]),
#     ([0.4, 0.4], [0, 1])
# ])

# test_data = np.array([
#     ([0.1, 0.1], 0),
#     ([0.2, 0.2], 1),
#     ([0.3, 0.3], 0),
#     ([0.4, 0.4], 1)
# ])
def create_output(i, n):
    output = np.zeros(n)
    output[i - 1] = 1
    return output

data_reader = DataReader("./data/classification/train/data.simple.train.100.csv")
xs = data_reader.get_columns(["x", "y"])
ys = data_reader.get_columns(["cls"])

train_data = [(x, create_output(int(y), 2)) for x, y in zip(xs, ys)]
test_data = [(x, y - 1) for x, y in zip(xs, ys)]

nn.train(train_data, 10, 100, 1, test_data)
