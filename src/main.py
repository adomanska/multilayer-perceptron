from neural_network.neural_network import NeuralNetwork
from activation_functions.sigmoid import Sigmoid
from activation_functions.re_lu import ReLU
import numpy as np
from data_reader import DataReader
from mnist_loader import MnistLoader

mnist_loader = MnistLoader('data/mnist')
training_data = mnist_loader.get_training_data()
print(len(training_data))
testing_data = mnist_loader.get_testing_data()
print(len(testing_data))

nn = NeuralNetwork()
nn.create_and_add_layer(784, 30, Sigmoid(), False)
nn.create_and_add_layer(30, 10, Sigmoid(), True)

nn.train(training_data[0:10000], 10, 100, 3, testing_data[0:1000])


# def create_output(i, n):
#     output = np.zeros(n)
#     output[i - 1] = 1
#     return output

# train_data_reader = DataReader("./data/classification/train/data.simple.train.1000.csv")
# xs = train_data_reader.get_columns(["x", "y"])
# ys = train_data_reader.get_columns(["cls"])
# train_data = [(x, create_output(int(y), 2)) for x, y in zip(xs, ys)]
# test_data_reader = DataReader("./data/classification/test/data.simple.test.1000.csv")
# xs = test_data_reader.get_columns(["x", "y"])
# ys = test_data_reader.get_columns(["cls"])
# test_data = [(x, y - 1) for x, y in zip(xs, ys)]

# nn.train(train_data, 10, 100, 0.5, test_data)
