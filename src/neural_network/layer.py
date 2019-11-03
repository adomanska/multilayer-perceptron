import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, input_count, neuron_count, activation_function, biases_enabled = True, weights = None, biases = None):
        self.input_count = input_count
        self.neuron_count = neuron_count
        self.activation_function = activation_function
        self.weights = weights if weights is not None else np.random.randn(neuron_count, input_count)
        self.weight_velocities = np.zeros((neuron_count, input_count))
        self.biases_enabled = biases_enabled
        if biases_enabled:
            self.bias_velocities = np.zeros(neuron_count)
            self.biases = biases if biases is not None else np.random.randn(neuron_count)
        self.weights_history = [self.weights]

    def activate(self, inputs):
        if self.biases_enabled:
            z = np.dot(self.weights, inputs) + self.biases
        else:
            z = np.dot(self.weights, inputs)
        activation = self.activation_function.calculate(z)
        self.last_z = z
        self.last_activation = activation
        self.last_input = inputs
        return activation

    @abstractmethod
    def backward_pass(self, y, delta):
        pass

    def update_weights_and_biases(self, nabla_w, nabla_b, eta, mini_batch_size, momentum = 0):
        self.weight_velocities = momentum * self.weight_velocities - (eta / mini_batch_size) * nabla_w
        self.weights = self.weights + self.weight_velocities
        if self.biases_enabled:
            self.bias_velocities = momentum * self.bias_velocities - (eta / mini_batch_size) * nabla_b
            self.biases = self.biases + self.bias_velocities

    def add_weights_to_history(self):
        self.weights_history.append(self.weights)
