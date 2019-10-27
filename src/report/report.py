from neural_network.regression_neural_network import RegressionNeuralNetwork
from neural_network.classification_neural_network import ClassificationNeuralNetwork

from activation_functions.sigmoid import Sigmoid
from activation_functions.re_lu import ReLU
from activation_functions.linear import Linear
from activation_functions.tan_h import TanH
from cost_functions.quadratic_cost import QuadraticCost
from data_transformations import create_test_data, create_train_data, ProblemType

import numpy as np

class Report:
    @staticmethod
    def test():
        batch_sizes = np.arange(1, 100, 10)
        epoch_counts = np.arange(100, 1500, 100)
        etas = np.arange(0.1, 1, 0.1)
        momenta = np.arange(0, 1, 0.1)

        cls_networks = Report.prepare_classification_networks()
        reg_networks = Report.prepare_regression_networks()

        train_data = create_train_data(ProblemType.Classification, "./data/classification/train/data.simple.train.1000.csv", ["x", "y"], ["cls"])
        test_data = create_test_data(ProblemType.Classification, "./data/classification/test/data.simple.test.1000.csv", ["x", "y"], ["cls"])

        f = open('results.txt', 'w')

        for nn in cls_networks:
            for epoch_count in epoch_counts:
                for eta in etas:
                    for momentum in momenta:
                        for batch_size in batch_sizes:
                            nn.train(train_data, batch_size, epoch_count, eta, momentum, test_data)
                            f.write('START TEST \n')
                            f.write(f'Epochs: {epoch_count}, eta: {eta}, momentum: {momentum}, batch size: {batch_size}\n')
                            # zaloguj nazwę zbioru i parametry sieci
                            f.write(f'Last accuracy: {nn.accuracies[-1]}\n')
                            f.write(f'Best accuracy: {np.max(nn.accuracies)}\n')
                            f.write('END TEST\n')
                            f.write('\n==================================================\n\n')
        
        f.close()
                

    @staticmethod
    def prepare_classification_networks():
        cls1 = ClassificationNeuralNetwork()
        cls1.create_and_add_layer(2, 10, Sigmoid(), False)
        cls1.create_and_add_layer(10, 15, Sigmoid(), False)
        cls1.create_and_add_layer(15, 2, Sigmoid(), True, QuadraticCost())

        cls2 = ClassificationNeuralNetwork()
        cls2.create_and_add_layer(2, 10, Sigmoid(), False)
        cls2.create_and_add_layer(10, 2, Sigmoid(), True, QuadraticCost())

        return [cls1, cls2]

    @staticmethod
    def prepare_regression_networks():
        reg1 = RegressionNeuralNetwork()
        reg1.create_and_add_layer(1, 9, Sigmoid(), False)
        reg1.create_and_add_layer(9, 1, Linear(), True)
        
        reg2 = RegressionNeuralNetwork()
        reg2.create_and_add_layer(1, 9, Sigmoid(), False)
        reg2.create_and_add_layer(9, 15, Sigmoid(), False)
        reg2.create_and_add_layer(15, 1, Linear(), True)

        return [reg1, reg2]
