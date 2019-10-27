from .neural_network import NeuralNetwork
import numpy as np

class ClassificationNeuralNetwork(NeuralNetwork):
    def evaluate(self, test_data):
        test_results = [(np.argmax(self._feed_forward(np.array(x))), y)
                        for (x, y) in test_data]
        return sum(int(x == np.argmax(y)) for (x, y) in test_results)

    def classify(self, test_data):
        return [(input, np.argmax(self._feed_forward(np.array(input)))) 
            for input in test_data]
        