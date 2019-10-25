from .output_layer import OutputLayer
from .hidden_layer import HiddenLayer
import random
import numpy as np
from abc import ABC, abstractmethod

class NeuralNetwork(ABC):
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def create_and_add_layer(self, input_count, neuron_count, activation_function, output = False, weights = None, biases = None):
        if output:
            layer = OutputLayer(input_count, neuron_count, activation_function, weights, biases)
        else:
            layer = HiddenLayer(input_count, neuron_count, activation_function, weights, biases)
        self.add_layer(layer)

    def _feed_forward(self, input):
        for layer in self.layers:
            input = layer.activate(input)

        return input

    def train(self, training_data, mini_batch_size, epochs_count, eta, momentum = 0, test_data = None):
        n_train = len(training_data)
        if test_data:
            n_test = len(test_data)
        
        for epoch in range(epochs_count):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self._update_mini_batch(mini_batch, eta, momentum)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))

    def _update_mini_batch(self, mini_batch, eta, momentum):
        mini_batch_size = len(mini_batch)
        nabla_b = np.array([np.zeros(layer.neuron_count) for layer in self.layers])
        nabla_w = np.array([np.zeros([layer.neuron_count, layer.input_count]) for layer in self.layers])
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = np.flip(self._backprop(x, y), 1)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        for layer, nb, nw in zip(self.layers, nabla_b, nabla_w):
            layer.update_weights_and_biases(nw, nb, eta, mini_batch_size, momentum)

    def _backprop(self, x, y):
        # feedforward
        self._feed_forward(x)
        # backward pass
        delta = None
        nabla_b = []
        nabla_w = []
        for i in range(1, len(self.layers) + 1):
            layer = self.layers[-i]
            delta, nw = layer.backward_pass(y, delta, self.layers[-i + 1].weights)
            nabla_b.append(delta)
            nabla_w.append(nw)
        return (nabla_b, nabla_w)

    @abstractmethod
    def evaluate(self, test_data):
        pass
