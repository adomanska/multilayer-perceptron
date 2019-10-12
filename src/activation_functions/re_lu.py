from .activation_function import ActivationFunction

class ReLU(ActivationFunction):
    def calculate(self, x):
        if x <= 0:
            return 0
        else:
            return x

    def calculate_derivative(self, x):
        if x <= 0:
            return 0
        else:
            return 1
