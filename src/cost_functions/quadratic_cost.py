from .cost_function import CostFunction
import numpy as np

class QuadraticCost(CostFunction):
	def calculate(self, a, y):
		return 0.5*np.linalg.norm(a-y)**2

	def calculate_derivative(self, z, a, y, activation_function):
		return (a-y) * activation_function.calculate_derivative(z)
