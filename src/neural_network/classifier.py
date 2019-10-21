from .neural_network import NeuralNetwork
import numpy as np

class Classifier(NeuralNetwork):
    def classify(self, test_data):
        return [(input, np.argmax(self._feed_forward(np.array(input)))) 
            for input in test_data]
