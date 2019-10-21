from data_reader import DataReader
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import itertools


class Visualisator:
    @staticmethod
    def draw_classifier_scatter_plot(classifier, test_data):
        # prepare grid points
        xs = np.arange(-1, 1, 0.01)
        grid_points = list(itertools.product(xs, xs))

        classification_results = classifier.classify(grid_points)

        # prepare points for the plot
        grouped_classification_results = Visualisator._group_by_class_id(
            classification_results)
        grouped_test_data = Visualisator._group_by_class_id(test_data)

        # Create the plot
        _, ax = plt.subplots(1)

        background_colors = ['#ff8080', '#99ff99', '#4d4dff']
        colors = ['red', 'green', 'blue']

        Visualisator._add_subplots(
            ax, grouped_classification_results, background_colors)
        Visualisator._add_subplots(ax, grouped_test_data, colors)

        plt.show()

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
