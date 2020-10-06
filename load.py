# Quick functions for loading in data.
# These will be imported and used in each model
import h5py
import os
import numpy as np

def load_in(path, key):
    # Load in x_test, x_train, y_test, y_train
    with h5py.File(path, 'r') as f:
        array = np.array(f[key], dtype='float64')
    return array # Might have to add .reshape

def train_test_set(train_path, test_path):
    x_train = load_in(train_path, 'mixture')
    y_train = load_in(train_path, 'target')
    x_test = load_in(test_path, 'mixture')
    y_test = load_in(test_path, 'target')
    return x_train, y_train, x_test, y_test

def main():
    train_path = os.path.abspath('data/Dev.hdf5')
    test_path = os.path.abspath('data/Test.hdf5')
    x_train, y_train, x_test, y_test = train_test_set(train_path, test_path)

if __name__ == '__main__':
    main()
