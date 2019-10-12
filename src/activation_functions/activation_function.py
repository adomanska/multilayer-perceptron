from abc import ABC, abstractmethod

class ActivationFunction(ABC):
	@abstractmethod
	def calculate(self, c):
		pass

	@abstractmethod
	def calculate_derivative(self, x):
		pass
