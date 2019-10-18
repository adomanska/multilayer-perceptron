from .layer import Layer
import numpy as np

class OutputLayer(Layer):
    def backward_pass(self, y, delta):
        d = self.cost_derivative(self.last_activation, y) * \
            self.activation_function.calculate_derivative(self.last_z)
        nabla_b = d
        nabla_w = np.dot(d, self.last_input)
        return nabla_b, nabla_w

    def cost_derivative(self, output_activations, y):
        print(output_activations-y)
        return (output_activations-y)
