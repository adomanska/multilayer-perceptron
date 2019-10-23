from .activation_function import ActivationFunction
import numpy as np

class Linear(ActivationFunction):
    def calculate(self, x):
        return x

    def calculate_derivative(self, x):
        return np.ones(x.shape)
