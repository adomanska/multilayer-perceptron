from .neural_network.neural_network import NeuralNetwork
from .activation_functions.sigmoid import Sigmoid
from .activation_functions.re_lu import ReLU
import numpy as np

nn = NeuralNetwork()
nn.create_and_add_layer(2, 2, Sigmoid())
nn.create_and_add_layer(2, 2, Sigmoid())
nn.create_and_add_layer(2, 2, Sigmoid(), True)

data = np.array([
    ([0.1, 0.1], 0),
    ([0.2, 0.2], 1),
    ([0.3, 0.3], 0),
    ([0.4, 0.4], 1)
])

test_data = np.array([
    ([0.1, 0.1], 0),
    ([0.2, 0.2], 1),
    ([0.3, 0.3], 0),
    ([0.4, 0.4], 1)
])

nn.train(data, 4, 1000, 0.001)
print(nn.evaluate(test_data))
