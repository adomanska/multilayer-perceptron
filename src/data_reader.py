import csv
import numpy as np

class DataReader:
    def __init__(self, path):
        with open(path, newline='') as data_file:
            reader = csv.reader(data_file, delimiter=',')
            headers = next(reader)
            self.data = {}
            for header in headers:
                self.data[header] = []
            for row in reader:
                for header, value in zip(headers, row):
                    self.data[header].append(value)

    def get_columns(self, column_names = None):
        if column_names is None:
            column_names = self.data.keys()
        return np.array([
            [float(x) for x in self.data[column]]
            for column in column_names
        ])
