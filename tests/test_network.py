import pytest
import numpy as np
from src.neural_network.classification_neural_network import ClassificationNeuralNetwork
from src.neural_network.hidden_layer import HiddenLayer
from src.neural_network.output_layer import OutputLayer
from src.activation_functions.activation_function import ActivationFunction

class SampleActivationFunc(ActivationFunction):
    def calculate(self, x):
        return x
    def calculate_derivative(self, x):
        return x

def test_feed_forward():
    inputs = np.array([0.8, 0.0, 0.0, 0.8])
    weights = np.array([
        [1/8, 1/8, 1/8, 1/8],
        [1/4, 1/4, 1/4, 1/4],
        [1/2, 1/2, 1/2, 1/2]
    ])
    biases = np.array([0.1, 0.1, 0.1])
    expected_result = np.array([1.7, 1.7])
    hidden_layer = HiddenLayer(4, 3, SampleActivationFunc(), weights=weights, biases=biases)
    output_layer = OutputLayer(3, 2, SampleActivationFunc(), weights=np.ones((2, 3)), biases=np.zeros(2))
    nn = ClassificationNeuralNetwork()
    nn.add_layer(hidden_layer)
    nn.add_layer(output_layer)
    result = nn._feed_forward(inputs)
    np.testing.assert_allclose(result, expected_result)
