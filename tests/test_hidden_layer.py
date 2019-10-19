import pytest
import numpy as np
from src.neural_network.hidden_layer import HiddenLayer
from src.activation_functions.sigmoid import Sigmoid

def test_backward_pass():
    sigmoid = Sigmoid()
    inputs = np.array([0.8, 0.0, 0.0, 0.8])
    weights = np.array([
        [1/8, 1/8, 1/8, 1/8],
        [1/4, 1/4, 1/4, 1/4],
        [1/2, 1/2, 1/2, 1/2]
    ])
    next_weights = np.array([
        [1/8, 1/8, 1/8],
        [1/4, 1/4, 1/4],
        [1/2, 1/2, 1/2]
    ])
    biases = np.array([0.1, 0.1, 0.1])
    layer = HiddenLayer(4, 3, sigmoid, weights, biases)
    delta = np.array([0.1, 0.2, 0.3])
    expected_nabla_b = np.array([
        0.2125 * sigmoid.calculate_derivative(0.3),
        0.2125 * sigmoid.calculate_derivative(0.5),
        0.2125 * sigmoid.calculate_derivative(0.9)
    ])
    expected_nabla_w = np.array([
        [expected_nabla_b[0] * el for el in inputs],
        [expected_nabla_b[1] * el for el in inputs],
        [expected_nabla_b[2] * el for el in inputs],
    ])

    layer.activate(inputs)
    nabla_b, nabla_w = layer.backward_pass(None, delta, next_weights)
    print(nabla_b)
    np.testing.assert_allclose(nabla_b, expected_nabla_b)
    np.testing.assert_allclose(nabla_w, expected_nabla_w)
