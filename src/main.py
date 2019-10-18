from neural_network.neural_network import NeuralNetwork
from activation_functions.sigmoid import Sigmoid
from activation_functions.re_lu import ReLU
import numpy as np

nn = NeuralNetwork()
nn.create_and_add_layer(2, 2, Sigmoid())
nn.create_and_add_layer(2, 2, Sigmoid())
nn.create_and_add_layer(2, 2, Sigmoid(), True)

data = np.array([
    ([1, 1], 1),
    ([2, 2], 2),
    ([3, 3], 1),
    ([4, 4], 2)
])

test_data = np.array([
    ([1, 1], 1),
    ([2, 2], 2),
    ([3, 3], 1),
    ([4, 4], 2)
])

nn.train(data, 4, 100, 1.5)
print(nn.evaluate(test_data))
