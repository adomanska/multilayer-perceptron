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

class Report:
    @staticmethod
    def run_tests():
        cls_networks = []#Report.prepare_classification_networks()
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
        train_filenames = ['data.simple.train.100.csv', 'data.simple.train.500.csv', 'data.three_gauss.train.100.csv', 'data.three_gauss.train.500.csv']
        test_filenames = [filename.replace("train", "test") for filename in train_filenames]

        print(train_filenames)

        for train_file, test_file in zip(train_filenames, test_filenames):
            train_data, output_count = create_train_data(ProblemType.Classification, train_dir + train_file, ["x", "y"], ["cls"])
            test_data, output_count = create_train_data(ProblemType.Classification, test_dir + test_file, ["x", "y"], ["cls"])
            cls_networks = Report.prepare_classification_networks(output_count)
            Report._test_networks(cls_networks, f'{output_directory}results.{train_file}', train_data, test_data)

    @staticmethod
    def test_regression():
        output_directory = './report/regression/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        train_dir = './data/regression/train/'
        test_dir = './data/regression/test/'
        train_filenames = ['data.activation.train.100.csv', 'data.activation.train.500.csv', 'data.cube.train.100.csv', 'data.cube.train.500.csv']
        test_filenames = [filename.replace("train", "test") for filename in train_filenames]

        print(train_filenames)

        for train_file, test_file in zip(train_filenames, test_filenames):
            train_data = create_train_data(ProblemType.Regression, train_dir + train_file, ["x"], ["y"])
            test_data = create_test_data(ProblemType.Regression, test_dir + test_file, ["x"], ["y"])
            reg_networks = Report.prepare_regression_networks()
            Report._test_networks(reg_networks, f'{output_directory}results.{train_file}', train_data, test_data)

    @staticmethod
    def _test_networks(networks, filename, train_data, test_data):
        batch_size = 50
        epoch_count = 100
        etas = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2]
        momenta = np.arange(0, 1, 0.25)

        f = open(filename, 'w')
        endl = '\n'
        
        # write headlines
        f.write('"eta,momentum"')
        for nn in networks:
            sizes = '-'.join([str(layer.neuron_count) for layer in nn.layers])
            f.write(f',{sizes}')
        f.write(endl)

        # train network and save results
        for eta in etas:
            for momentum in momenta:
                f.write(f'"{eta},{momentum}"')

                print("eta={0} momentum={1}".format(eta, momentum))
                for nn in networks:
                    nn.train(train_data, batch_size, epoch_count, eta, momentum, test_data)
                    f.write(f',{np.max(nn.accuracies)}')
                
                f.write(endl)
        
        f.close()

    @staticmethod
    def prepare_classification_networks(outputs_count):
        networks = []
        neurons_confs = [(2, 1), (2, 5), (2, 10), (2, 1, 5), (2, 10, 5), (2, 1, 5, 5), (2, 5, 10, 1), (2, 5, 5, 5, 5), (2, 1, 5, 10, 5)]

        for conf in neurons_confs:
            nn = ClassificationNeuralNetwork()
            for i in range(1, len(conf)):
                nn.create_and_add_hidden_layer(conf[i - 1], conf[i], Sigmoid())
            nn.create_and_add_output_layer(conf[-1], outputs_count, Sigmoid(), QuadraticCost())
            networks.append(nn)

        return networks

    @staticmethod
    def prepare_regression_networks():
        networks = []
        neurons_confs = [ (1, 5), (1, 10), (1, 1, 5), (1, 10, 5), (1, 1, 5, 5), (1, 5, 10, 1), (1, 5, 5, 5, 5), (1, 1, 5, 10, 5)]

        for conf in neurons_confs:
            nn = RegressionNeuralNetwork()
            for i in range(1, len(conf)):
                nn.create_and_add_hidden_layer(conf[i - 1], conf[i], Sigmoid())
            nn.create_and_add_output_layer(conf[-1], 1, Linear(), CrossEntropyCost())
            networks.append(nn)

        return networks