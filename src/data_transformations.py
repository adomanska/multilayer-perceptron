from enum import Enum
from data_reader import DataReader
import numpy as np

class ProblemType(Enum):
    Classification = "CLASSIFICATION",
    Regression = "REGRESSION"

def create_classification_output(i, n):
    output = np.zeros(n)
    output[i - 1] = 1
    return output

def create_train_data(problem_type, data_path, x_cols, y_cols):
    if isinstance(y_cols, list) and len(y_cols) > 1 and problem_type == ProblemType.Classification:
        raise ValueError("There should be one class per record")

    data_reader = DataReader(data_path)
    xs = data_reader.get_columns(x_cols)
    ys = data_reader.get_columns(y_cols)

    if problem_type == ProblemType.Classification:
        max_class = int(max(ys))
        return [(x, create_classification_output(int(y), max_class)) for x, y in zip(xs, ys)], max_class
    else:
        return [(x, y) for x, y in zip(xs, ys)]

def create_test_data(problem_type, data_path, x_cols, y_cols):
    data_reader = DataReader(data_path)
    xs = data_reader.get_columns(x_cols)
    ys = data_reader.get_columns(y_cols)

    if problem_type == ProblemType.Classification:
        return [(x, y[0] - 1) for x, y in zip(xs, ys)] # y[0] because classification output always has one element
    else:
        return [(x, y) for x, y in zip(xs, ys)]
