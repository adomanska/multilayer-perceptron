import pytest
import numpy as np
from src.neural_network.hidden_layer import HiddenLayer
from src.activation_functions.activation_function import ActivationFunction

class SampleActivationFunc(ActivationFunction):
    def calculate(self, x):
        return x
    def calculate_derivative(self, x):
        return x

def test_activate():
    inputs = np.array([0.8, 0.0, 0.0, 0.8])
    weights = np.array([
        [1/8, 1/8, 1/8, 1/8],
        [1/4, 1/4, 1/4, 1/4],
        [1/2, 1/2, 1/2, 1/2]
    ])
    biases = np.array([0.1, 0.1, 0.1])
    expected_result = np.array([0.3, 0.5, 0.9])
    layer = HiddenLayer(4, 3, SampleActivationFunc(), weights=weights, biases=biases)
    result = layer.activate(inputs)
    print(result, expected_result)
    np.testing.assert_allclose(result, expected_result)
