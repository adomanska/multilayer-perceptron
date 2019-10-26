from .cost_function import CostFunction
import numpy as np

class CrossEntropyCost(CostFunction):
	def calculate(self, a, y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

	def calculate_derivative(self, z, a, y, activation_function):
		return (a-y)
