from .activation_function import ActivationFunction
import numpy as np

class TanH(ActivationFunction):
    def calculate(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def calculateDerivative(self, x):
        return 1 - self.calculate(x) ^ 2
