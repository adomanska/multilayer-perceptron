from neural_network.regression_neural_network import RegressionNeuralNetwork
from neural_network.classification_neural_network import ClassificationNeuralNetwork

import numpy as np
import os
import sys

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
            f.write(f'Architecture,Type,Best accuracy,Best weights{endl}')

        arch = '_'.join([f'{layer.neuron_count}' for layer in nn.layers])

        task_type = 'classification' if(type(nn).__name__ == 'ClassificationNeuralNetwork') else 'regression'

        best_acc_index = np.argmax(nn.accuracies)
        acc = round(nn.accuracies[best_acc_index]*100, ndigits = 2)

        # prevent numpy from truncating arrays
        np.set_printoptions(threshold=sys.maxsize)
        best_weights = '; '.join([f'{layer.weights_history[best_acc_index]}' for layer in nn.layers]).replace('\n', '')

        f.write(f'{arch},{task_type},{acc}%,"{best_weights}"{endl}')

        f.close()

