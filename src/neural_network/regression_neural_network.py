from .neural_network import NeuralNetwork
import numpy as np

class RegressionNeuralNetwork(NeuralNetwork):
    def evaluate(self, test_data):
        test_results = [(self._feed_forward(np.array(x)), y)
                        for (x, y) in test_data]
        return sum(int(abs(x - y) < 1) for (x, y) in test_results)
