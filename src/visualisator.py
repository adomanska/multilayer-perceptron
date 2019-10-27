from data_reader import DataReader
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import itertools


class Visualisator:
    @staticmethod
    def visualise_classification(nn, test_data):
        Visualisator.draw_loss_function(nn)
        Visualisator.draw_classifier_scatter_plot(nn, test_data)
        Visualisator.draw_weights_history(nn)
        plt.show()

    @staticmethod
    def visualise_regression(nn, test_data):
        Visualisator.draw_loss_function(nn)
        Visualisator.draw_regression_results(nn, test_data)
        Visualisator.draw_weights_history(nn)
        plt.show()

    @staticmethod
    def draw_classifier_scatter_plot(classifier, test_data):
        # prepare grid points
        xs = np.arange(-1, 1, 0.01)
        grid_points = list(itertools.product(xs, xs))

        classification_results = classifier.classify(grid_points)

        # prepare points for the plot
        grouped_classification_results = \
            Visualisator._group_by_class_id(classification_results)
        
        grouped_test_data = Visualisator._group_by_class_id([(x, np.argmax(y)) for (x, y) in test_data])

        # Create the plot
        _, ax = plt.subplots(1)

        background_colors = ['#ff8080', '#99ff99', '#4d4dff']
        colors = ['red', 'green', 'blue']

        Visualisator._add_subplots(
            ax, grouped_classification_results, background_colors)
        Visualisator._add_subplots(ax, grouped_test_data, colors)

        plt.title('Test set points vs predicted sets\' shapes' )

    @staticmethod
    def draw_regression_results(nn, test_data):
        sorted_data = sorted(test_data, key=lambda x: x[0][0])
        xs = np.array([x[0] for x, y in sorted_data])
        ys = np.array([y[0] for x, y in sorted_data])

        prediction_results = nn.predict(xs)

        y_predicted = list(map(itemgetter(1), prediction_results))

        plt.figure()
        plt.plot(xs, ys, c = 'red')
        plt.plot(xs, y_predicted, c = 'blue')
        plt.legend(['test set values', 'predicted values'])
        plt.xlabel('x')
        plt.ylabel('y')

    @staticmethod
    def draw_weights_history(nn):
        # reduce plot margins
        plt.figure()
        plt.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)

        for layer_index, layer in enumerate(nn.layers):
            for prev_layer_neuron_index in range(layer.input_count):
                # get weights of edges from one neuron from previous layer to all neurons in current layer 
                weights = np.transpose([historical_weights[:,prev_layer_neuron_index] for historical_weights in layer.weights_history])
                for neuron_index in range(layer.neuron_count):
                    rows_count = layer.input_count
                    cols_count =  len(nn.layers)
                    current_plot_index = len(nn.layers) * prev_layer_neuron_index + layer_index + 1

                    ax = plt.subplot(rows_count, cols_count, current_plot_index)
                    plt.plot(range(len(weights[neuron_index])), weights[neuron_index])
                    
                    if(prev_layer_neuron_index == 0):
                        plt.title(f'Weights of edges between layer {layer_index} and {layer_index + 1}') # show title only above the top chart in each column
                    plt.setp(ax.get_xticklabels(), visible = (prev_layer_neuron_index == layer.input_count -1)) # show x axis tick labels only on the bottom chart in each column
                    ax.set_ylabel('Weight', fontdict = { 'size': 9 })

            plt.legend([f'{x}' for x in range(layer.neuron_count)]).draggable() # show legend only for the bottom chart in each column
            ax.set_xlabel('Epoch')

    @staticmethod
    def draw_loss_function(nn):
        ys_train = nn.epoch_train_costs
        ys_test = nn.epoch_test_costs
        xs = range(len(ys_train))

        plt.figure()
        plt.plot(xs, ys_train)
        plt.plot(xs, ys_test)
        plt.legend(['train set loss', 'test set loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

    @staticmethod
    def _group_by_class_id(data):
        group_ids = set(map(itemgetter(1), data))
        grouped_points = [
            {
                "id": group_id,
                "xs": [elem[0][0] for elem in data if elem[1] == group_id],
                "ys": [elem[0][1] for elem in data if elem[1] == group_id]
            }
            for group_id in group_ids]

        return grouped_points

    @staticmethod
    def _add_subplots(ax, grouped_points, colors):
        for group, color in zip(grouped_points, colors):
            ax.scatter(group['xs'], group['ys'], c=color, edgecolors='none')
