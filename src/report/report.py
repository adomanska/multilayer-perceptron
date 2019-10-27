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
                f.write(f'{eta},{momentum}"')

                print("eta={0} momentum={1}".format(eta, momentum))
                for nn in networks:
                    nn.train(train_data, batch_size, epoch_count, eta, momentum, test_data)
                    f.write(f',{np.max(nn.accuracies)}')
                
                f.write(endl)
        
        f.close()

    @staticmethod
    def prepare_classification_networks(outputs_count):
        networks = []
        # no hidden layers
        # cls1 = ClassificationNeuralNetwork()
        # cls1.create_and_add_output_layer(2, outputs_count, Sigmoid(), QuadraticCost())
        # networks.append(cls1)

        # 1 hidden layer
        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 1, Sigmoid())
        cls1.create_and_add_output_layer(1, outputs_count, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 5, Sigmoid())
        cls1.create_and_add_output_layer(5, outputs_count, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 10, Sigmoid())
        cls1.create_and_add_output_layer(10, outputs_count, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        # # 2 hidden
        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 1, Sigmoid())
        cls1.create_and_add_hidden_layer(1, 5, Sigmoid())
        cls1.create_and_add_output_layer(5, outputs_count, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 1, Sigmoid())
        cls1.create_and_add_hidden_layer(1, 5, Sigmoid())
        cls1.create_and_add_output_layer(5, outputs_count, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        # 3 hidden
        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 1, Sigmoid())
        cls1.create_and_add_hidden_layer(1, 5, Sigmoid())
        cls1.create_and_add_hidden_layer(5, 5, Sigmoid())
        cls1.create_and_add_output_layer(5, outputs_count, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 5, Sigmoid())
        cls1.create_and_add_hidden_layer(5, 10, Sigmoid())
        cls1.create_and_add_hidden_layer(10, 1, Sigmoid())
        cls1.create_and_add_output_layer(1, outputs_count, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        # 4 hidden
        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 5, Sigmoid())
        cls1.create_and_add_hidden_layer(5, 5, Sigmoid())
        cls1.create_and_add_hidden_layer(5, 5, Sigmoid())
        cls1.create_and_add_hidden_layer(5, 5, Sigmoid())
        cls1.create_and_add_output_layer(5, outputs_count, Sigmoid(), QuadraticCost())
        networks.append(cls1)

        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_hidden_layer(2, 1, Sigmoid())
        cls1.create_and_add_hidden_layer(1, 5, Sigmoid())
        cls1.create_and_add_hidden_layer(5, 10, Sigmoid())
        cls1.create_and_add_hidden_layer(10, 5, Sigmoid())
        cls1.create_and_add_output_layer(5, outputs_count, Sigmoid(), QuadraticCost())
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