from neural_network.regression_neural_network import RegressionNeuralNetwork
from neural_network.classification_neural_network import ClassificationNeuralNetwork

from activation_functions.sigmoid import Sigmoid
from activation_functions.re_lu import ReLU
from activation_functions.linear import Linear
from activation_functions.tan_h import TanH
from cost_functions.quadratic_cost import QuadraticCost
from cost_functions.cross_entropy_cost import CrossEntropyCost
from data_transformations import create_test_data, create_train_data, ProblemType

import numpy as np
import os

class FileWriter:
    @staticmethod
    def save_results(nn):
        output_directory = './results/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        filename = 'results.csv'
        filepath = f'{output_directory}{filename}'

        f = open(filepath, 'a+')
        endl = '\n'

        # if file doesn't exist or is empty, write headers
        if(not os.path.isfile(filepath) or os.path.getsize(filepath) == 0):
            f.write(f'Architecture,Type,Best accuracy,Seed{endl}')

        arch = '_'.join([f'{layer.neuron_count}' for layer in nn.layers])
        task_type = 'classification' if(type(nn).__name__ == 'ClassificationNeuralNetwork') else 'regression'
        acc = round(np.max(nn.accuracies)*100, ndigits = 2)
        seed = f'"{nn.seed}"'.replace('\n', '')

        f.write(f'{arch},{task_type},{acc}%,{seed}{endl}')

        f.close()

