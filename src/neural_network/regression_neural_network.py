from .neural_network import NeuralNetwork
import numpy as np

class RegressionNeuralNetwork(NeuralNetwork):
    def evaluate(self, test_data):
        test_results = [(self._feed_forward(np.array(x)), y)
                        for (x, y) in test_data]
        return sum([((x - y)**2)/2 for (x, y) in test_results])

    def predict(self, test_data):
        return [(x, self._feed_forward(np.array([x]))[0]) for x in test_data]

