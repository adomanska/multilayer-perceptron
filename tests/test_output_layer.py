import pytest
import numpy as np
from src.neural_network.output_layer import OutputLayer
from src.activation_functions.sigmoid import Sigmoid
from src.cost_functions.quadratic_cost import QuadraticCost

def test_backward_pass():
    sigmoid = Sigmoid()
    mse = QuadraticCost()
    inputs = np.array([0.8, 0.0, 0.0, 0.8])
    weights = np.array([
        [1/8, 1/8, 1/8, 1/8],
        [1/4, 1/4, 1/4, 1/4],
        [1/2, 1/2, 1/2, 1/2]
    ])
    biases = np.array([0.1, 0.1, 0.1])
    layer = OutputLayer(4, 3, sigmoid, mse, weights, biases)
    y = np.array([0, 1, 0])
    expected_nabla_b = np.array([
        (sigmoid.calculate(0.3) - 0) * sigmoid.calculate_derivative(0.3),
        (sigmoid.calculate(0.5) - 1) * sigmoid.calculate_derivative(0.5),
        (sigmoid.calculate(0.9) - 0) * sigmoid.calculate_derivative(0.9)
    ])
    expected_nabla_w = np.array([
        [expected_nabla_b[0] * el for el in inputs],
        [expected_nabla_b[1] * el for el in inputs],
        [expected_nabla_b[2] * el for el in inputs],
    ])

    layer.activate(inputs)
    nabla_b, nabla_w = layer.backward_pass(y, None)
    np.testing.assert_allclose(nabla_b, expected_nabla_b)
    np.testing.assert_allclose(nabla_w, expected_nabla_w)
