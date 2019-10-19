from src.data_reader import DataReader
import numpy as np 
import pytest

def test_get_all_columns():
    expected_result = np.array([
        [-0.1,0.7,1],
        [0.5,-0.7,2],
        [0.8,-0.2,1]
    ])
    data_reader = DataReader("./tests/test_data.csv")
    data = data_reader.get_columns()

    np.testing.assert_array_equal(data, expected_result)

def test_get_specified_columns():
    expected_result = np.array([
        [-0.1,1],
        [0.5,2],
        [0.8,1]
    ])
    data_reader = DataReader("./tests/test_data.csv")
    data = data_reader.get_columns(['x', 'cls'])

    np.testing.assert_array_equal(data, expected_result)
