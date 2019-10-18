from neural_network import NeuralNetwork
from activation_functions.sigmoid import Sigmoid

nn = NeuralNetwork()
nn.create_and_add_layer(2, 2, Sigmoid())
nn.create_and_add_layer(2, 2, Sigmoid())
nn.create_and_add_layer(2, 2, Sigmoid())

data = [
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8]
]

nn.train(data, 2, 2, 0.1)
