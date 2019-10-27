from neural_network.regression_neural_network import RegressionNeuralNetwork
from neural_network.classification_neural_network import ClassificationNeuralNetwork

from activation_functions.sigmoid import Sigmoid
from activation_functions.re_lu import ReLU
from activation_functions.linear import Linear
from activation_functions.tan_h import TanH
from cost_functions.quadratic_cost import QuadraticCost
from data_transformations import create_test_data, create_train_data, ProblemType

import numpy as np
import os

class Report:
    @staticmethod
    def run_tests():
        cls_networks = Report.prepare_classification_networks()
        reg_networks = Report.prepare_regression_networks()

        train_data = create_train_data(ProblemType.Classification, "./data/classification/train/data.simple.train.1000.csv", ["x", "y"], ["cls"])
        test_data = create_train_data(ProblemType.Classification, "./data/classification/test/data.simple.test.1000.csv", ["x", "y"], ["cls"])

        Report._test_networks(cls_networks, '/report/classification/classification.txt', train_data, test_data)

    @staticmethod
    def test_classification():
        output_directory = './report/classification/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        train_dir = './data/classification/train/'
        test_dir = './data/classification/test/'
        (_, _, train_filenames) = os.walk(train_dir).__next__()
        (_, _, test_filenames) = os.walk(test_dir).__next__()

        print(train_filenames)

        cls_networks = Report.prepare_classification_networks()

        for train_file, test_file in zip(train_filenames, test_filenames):
            train_data = create_train_data(ProblemType.Classification, train_dir + train_file, ["x", "y"], ["cls"])
            test_data = create_train_data(ProblemType.Classification, test_dir + test_file, ["x", "y"], ["cls"])

            Report._test_networks(cls_networks, f'{output_directory}{train_file}/', train_data, test_data)

    @staticmethod
    def _test_networks(networks, output_dir, train_data, test_data):
        batch_sizes = np.arange(10, 30, 10)
        epoch_counts = np.arange(100, 300, 100)
        etas = np.arange(0.1, 0.3, 0.1)
        momenta = np.arange(0, 0.3, 0.1)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for index, nn in enumerate(networks):
            f = open(f'{output_dir}/{index}.txt', 'w')

            for epoch_count in epoch_counts:
                for eta in etas:
                    for momentum in momenta:
                        for batch_size in batch_sizes:
                            nn.train(train_data, batch_size, epoch_count, eta, momentum, test_data)
                            f.write('START TEST \n')
                            f.write(f'Epochs: {epoch_count}, eta: {eta}, momentum: {momentum}, batch size: {batch_size}\n')
                            f.write(f'{type(nn).__name__}\n')
                            f.write('Layers:\n')
                            for layer in nn.layers:
                                f.write(f'neurons: {layer.neuron_count}, activation: {type(layer.activation_function).__name__}\n')
                            
                            f.write(f'cost function: {type(nn.layers[-1].cost_function).__name__}\n')
                            f.write(f'Last accuracy: {nn.accuracies[-1]}\n')
                            f.write(f'Best accuracy: {np.max(nn.accuracies)}\n')
                            f.write('END TEST\n')
                            f.write('\n==================================================\n\n')
            f.close()

    @staticmethod
    def prepare_classification_networks():
        networks = []
        # no hidden layers
        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_output_layer(2, 2, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        # 1 hidden layer
        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 1, Sigmoid())
        cls1.create_and_add_output_layer(1, 2, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 5, Sigmoid())
        cls1.create_and_add_output_layer(5, 2, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 10, Sigmoid())
        cls1.create_and_add_output_layer(10, 2, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        # 2 hidden
        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 1, Sigmoid())
        cls1.create_and_add_hidden_layer(1, 5, Sigmoid())
        cls1.create_and_add_output_layer(5, 2, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        return networks

    @staticmethod
    def prepare_regression_networks():
        reg1 = RegressionNeuralNetwork()
        reg1.create_and_add_hidden_layer(1, 9, Sigmoid())
        reg1.create_and_add_output_layer(9, 1, Linear(), QuadraticCost())
        
        reg2 = RegressionNeuralNetwork()
        reg2.create_and_add_hidden_layer(1, 9, Sigmoid())
        reg2.create_and_add_hidden_layer(9, 15, Sigmoid())
        reg2.create_and_add_output_layer(15, 1, Linear(), QuadraticCost())

        return [reg1, reg2]