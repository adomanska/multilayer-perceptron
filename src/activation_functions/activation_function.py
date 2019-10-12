from abc import ABC, abstractmethod

class ActivationFunction(ABC):
	@abstractmethod
	def calculate(self, c):
		pass

	@abstractmethod
	def calculateDerivative(self, x):
		pass
