from .activation_function import ActivationFunction
import numpy as np

class Sigmoid(ActivationFunction):
    def calculate(self, x):
        return 1.0/(1.0+np.exp(-x))

    def calculateDerivative(self, x):
        return self.calculate(x)*(1-self.calculate(x))
