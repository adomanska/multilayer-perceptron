from .layer import Layer
import numpy as np

class OutputLayer(Layer):
    def __init__(self, input_count, neuron_count, activation_function, cost_function, biases_enabled = True, weights = None, biases = None):
        super(OutputLayer, self).__init__(input_count, neuron_count, activation_function, biases_enabled, weights, biases)
        self.cost_function = cost_function

    def backward_pass(self, y, delta, next_weights = None):
        d = self.cost_function.calculate_derivative(self.last_z, self.last_activation, y, self.activation_function)
        nabla_b = d
        nabla_w = np.dot(np.array([d]).transpose(), np.array([self.last_input]))
        return nabla_b, nabla_w
