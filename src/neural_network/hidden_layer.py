from .layer import Layer
import numpy as np

class HiddenLayer(Layer):
    def backward_pass(self, y, delta, next_weights = None):
        sp = self.activation_function.calculate_derivative(self.last_z)
        delta = np.dot(next_weights.transpose(), delta) * sp
        nabla_b = delta
        nabla_w = np.dot(np.array([delta]).transpose(), np.array([self.last_input]))
        return nabla_b, nabla_w
