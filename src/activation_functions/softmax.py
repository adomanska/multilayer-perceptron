from .activation_function import ActivationFunction
import numpy as np

class Softmax(ActivationFunction):
    def calculate(self, x):
        x_stable = x - np.max(x, axis=0)
        return np.exp(x_stable) / np.sum(np.exp(x_stable), axis=0, keepdims=True)

    def calculate_derivative(self, x):
        values = self.calculate(x)
        return values * (1 - values)
