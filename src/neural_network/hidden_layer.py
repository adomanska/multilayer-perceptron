from .layer import Layer
import numpy as np

class HiddenLayer(Layer):
    def backward_pass(self, y, delta):
        sp = self.activation_function.calculate_derivative(self.last_z)
        delta = np.dot(self.weights.transpose(), delta) * sp
        nabla_b = delta
        nabla_w = np.dot(delta, np.array(self.last_input).transpose())
        return nabla_b, nabla_w
