from layer import Layer

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