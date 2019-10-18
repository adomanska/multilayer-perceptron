import numpy as np

class Layer:
    def __init__(self, input_count, neuron_count, activation_function, weights = None, biases = None):
        self.input_count = input_count
        self.neuron_count = neuron_count
        self.activation_function = activation_function
        self.weights = weights if weights is not None else np.random.rand(input_count, neuron_count)
        self.biases = biases if biases is not None else np.random.rand(neuron_count)

    def activate(self, input):
        return self.activation_function(np.dot(input, self.weights) + self.biases)
