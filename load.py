'''
To use these functions are simple. In fact, you only need train_test_set,
the path to the training data, and the path to the test data.

from load import train_test_set as tts

train_path = 'path/to/train/'
test_path = 'path/to/test/'
x_train, y_train, x_test, y_test = tts(train_path, test_path)
'''
import h5py
import os
import numpy as np

def load_in(path, key):
    '''
    Load in data from .hdf5 files formed from preprocess.py

    Parameters:
    ------------
    - path (str):
    Path of hdf5 file.

    - key (str):
    Key of hdf5 file. Should be either mixture or target.

    Returns:
    ---------
    Loaded array of shape relevant to neural networks.
    '''
    with h5py.File(path, 'r') as f:
        array = np.array(f[key], dtype='float32')
    return np.expand_dims(array, axis=3)

def train_test_set(train_path, test_path):
    '''
    Load the train and test variables.

    Parameters:
    -----------
    - train_path (str):
        Path of hdf5 file containing training data.
    - test_path (str):
        Path of hdf5 file containing testing data.

    Returns:
    --------
    x_train, y_train, x_test, y_test
    '''
    x_train = load_in(train_path, 'mixture')
    y_train = load_in(train_path, 'target')
    x_test = load_in(test_path, 'mixture')
    y_test = load_in(test_path, 'target')
    return x_train, y_train, x_test, y_test
