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
    layer = HiddenLayer(4, 3, SampleActivationFunc(), weights, biases)
    result = layer.activate(inputs)
    np.testing.assert_allclose(result, expected_result)

def test_update_weights_and_biases():
    initial_weights = np.array([
        [0.4, 0.5, 0.6],
        [0.8, 0.7, 0.6]
    ])
    initial_biases = np.array([0.1, 0.2])
    nabla_w = np.array([
        [1, 2, 3],
        [3, 2, 1]
    ])
    nabla_b = np.array([0.5, 1])
    eta = 1
    mini_batch_size = 10
    expected_weights = np.array([
        [0.3, 0.3, 0.3],
        [0.5, 0.5, 0.5]
    ])
    expected_biases = np.array([0.05, 0.1])
    layer = HiddenLayer(3, 2, SampleActivationFunc(), initial_weights, initial_biases)

    print(layer.weights)
    layer.update_weights_and_biases(nabla_w, nabla_b, eta, mini_batch_size)

    print(layer.weights)
    np.testing.assert_allclose(layer.weights, expected_weights)
    np.testing.assert_allclose(layer.biases, expected_biases)
