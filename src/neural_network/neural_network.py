from .output_layer import OutputLayer
from .hidden_layer import HiddenLayer
import random
import numpy as np
from abc import ABC, abstractmethod

class NeuralNetwork(ABC):
    def __init__(self, seed = 123):
        np.random.seed(seed)
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def create_and_add_hidden_layer(
        self,
        input_count, 
        neuron_count, 
        activation_function,
        biases_enabled = True,
        weights = None, 
        biases = None
    ):
        self.add_layer(HiddenLayer(input_count, neuron_count, activation_function, weights, biases))

    def create_and_add_output_layer(
        self,
        input_count, 
        neuron_count, 
        activation_function,
        cost_function,
        biases_enabled = True,
        weights = None, 
        biases = None
    ):
        self.add_layer(OutputLayer(input_count, neuron_count, activation_function, cost_function, weights, biases))

    def _feed_forward(self, input):
        for layer in self.layers:
            input = layer.activate(input)

        return input

    def train(self, training_data, mini_batch_size, epochs_count, eta, momentum = 0, test_data = None):
        n_train = len(training_data)
        self.accuracies = []
        self.epoch_train_costs = []
        self.epoch_test_costs = []

        if test_data:
            n_test = len(test_data)
        
        for epoch in range(epochs_count):
            self.summed_train_cost = 0
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self._update_mini_batch(mini_batch, eta, momentum)
            if test_data:
                good_predictions_count = self.evaluate(test_data)
                self.accuracies.append(good_predictions_count / n_test)
                print("Epoch {0}: {1} / {2}".format(epoch, good_predictions_count , n_test))
                self._calculate_test_set_cost(test_data)
            else:
                print("Epoch {0} complete".format(epoch))
            for layer in self.layers:
                layer.add_weights_to_history()
                    
            # save average cost
            avg_train_cost = self.summed_train_cost / n_train
            self.epoch_train_costs.append(avg_train_cost)
            

    def _update_mini_batch(self, mini_batch, eta, momentum):
        mini_batch_size = len(mini_batch)
        nabla_b = np.array([np.zeros(layer.neuron_count) for layer in self.layers])
        nabla_w = [np.zeros((layer.neuron_count, layer.input_count)) for layer in self.layers]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self._backprop(x, y)
            delta_nabla_b.reverse()
            delta_nabla_w.reverse()
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        for layer, nb, nw in zip(self.layers, nabla_b, nabla_w):
            layer.update_weights_and_biases(nw, nb, eta, mini_batch_size, momentum)

    def _backprop(self, x, y):
        # feedforward
        result = self._feed_forward(x)
        cost = self.layers[-1].cost_function.calculate(result, y)
        self.summed_train_cost += cost
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

    def _calculate_test_set_cost(self, test_data):
        cost_sum = 0
        test_size = len(test_data)

        for (x, y) in test_data:
            result = self._feed_forward(x)
            cost_sum += self.layers[-1].cost_function.calculate(np.array(result), np.array(y)).sum()
        
        self.epoch_test_costs.append(cost_sum / test_size)

    @abstractmethod
    def evaluate(self, test_data):
        pass
