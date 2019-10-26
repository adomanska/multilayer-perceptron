from abc import ABC, abstractmethod

class CostFunction(ABC):
	@abstractmethod
	def calculate(self, a, y):
		pass

	@abstractmethod
	def calculate_derivative(self, z, a, y, activation_function):
		pass
