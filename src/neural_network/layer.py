import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, input_count, neuron_count, activation_function, weights = None, biases = None):
        self.input_count = input_count
        self.neuron_count = neuron_count
        self.activation_function = activation_function
        self.weights = weights if weights is not None else np.random.rand(neuron_count, input_count)
        self.biases = biases if biases is not None else np.random.rand(neuron_count)

    def activate(self, inputs):
        z = np.dot(self.weights, inputs) + self.biases
        activation = self.activation_function.calculate(z)
        self.last_z = z
        self.last_activation = activation
        self.last_input = inputs
        return activation

    @abstractmethod
    def backward_pass(self, y, delta):
        pass

    def update_weights_and_biases(self, nabla_w, nabla_b, eta, mini_batch_size):
        self.weights = self.weights - (eta / mini_batch_size) * nabla_w
        self.biases = self.biases - (eta / mini_batch_size) * nabla_b
