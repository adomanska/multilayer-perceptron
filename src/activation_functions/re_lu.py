from .activation_function import ActivationFunction
import numpy as np

class ReLU(ActivationFunction):
    def calculate(self, x):
        return np.array([calc_ReLU(e) for e in x])

    def calculate_derivative(self, x):
        return np.array([calc_ReLU_der(e) for e in x])

def calc_ReLU(x):
    if x <= 0:
        return 0
    else:
        return x

def calc_ReLU_der(x):
    if x <= 0:
        return 0
    else:
        return 1