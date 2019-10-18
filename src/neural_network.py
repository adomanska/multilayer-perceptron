from layer import Layer
import random
import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def create_and_add_layer(self, input_count, neuron_count, activation_function, weights = None, biases = None):
        layer = Layer(input_count, neuron_count, activation_function, weights, biases)
        self.add_layer(layer)

    def feed_forward(self, input):
        for layer in self.layers:
            input = layer.activate(input)

        return input

    def train(self, training_data, mini_batch_size, epochs_count, eta, test_data = None):
        n_train = len(training_data)
        if test_data:
            n_test = len(test_data)
        
        for epoch in range(epochs_count):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))

    def update_mini_batch(self, mini_batch, eta):
        pass

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
