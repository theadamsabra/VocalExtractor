'''
To use these functions are simple. In fact, you only need train_test_set,
the path to the training data, and the path to the test data.

from load import train_test_set as tts
x_train, y_train, x_test, y_test = tts(train_path, test_path)

It's that simple.
'''
import h5py
import os
import numpy as np

def load_in(path, key):
    # Load in data from .hdf5 files
    with h5py.File(path, 'r') as f:
        array = np.array(f[key], dtype='float32')
    return array.reshape(array.shape[0], array.shape[2], array.shape[1], 1)

def train_test_set(train_path, test_path):
    # Load the train and test variables
    x_train = load_in(train_path, 'mixture')
    y_train = load_in(train_path, 'target')
    x_test = load_in(test_path, 'mixture')
    y_test = load_in(test_path, 'target')
    return x_train, y_train, x_test, y_test
