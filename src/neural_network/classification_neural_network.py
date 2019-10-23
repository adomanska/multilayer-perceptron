from .neural_network import NeuralNetwork
import numpy as np

class ClassificationNeuralNetwork(NeuralNetwork):
    def evaluate(self, test_data):
        test_results = [(np.argmax(self._feed_forward(np.array(x))), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        