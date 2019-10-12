from .activation_function import ActivationFunction

class LeakyReLU(ActivationFunction):
    def calculate(self, x):
        if x < 0:
            return 0.01 * x
        else:
            return x

    def calculateDerivative(self, x):
        if x < 0:
            return 0.01
        else:
            return 1
