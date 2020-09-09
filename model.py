import h5py
import numpy as np
import os

# Functions for bringing in the test/train data
def load_in(path, key):
    # Load in x_test, x_train, y_test, y_train
    with h5py.File(path, 'r') as f:
        array = np.array(f[key], dtype = 'float64')
    return array # Might have to add .reshape

def train_test_set(train_path, test_path):
    x_train = load_in(train_path, 'mixture')
    y_train = load_in(train_path, 'target')
    x_test = load_in(test_path, 'mixture')
    y_test = load_in(test_path, 'target')
    return x_train, y_train, x_test, y_test

# Custom loss function for VocalUNet
def u_net_loss():
    pass

class VocalUNet:
    def __init__(self, batch_size, num_classes, epochs):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
    # Function to train the model
    def train_model(self, x_train, y_train):
        pass
    # Function to test the model
    def test_model(self, x_test, y_test):
        pass
    # Evaluation of model using specific evaluation criteria.
    def eval(self):
        pass

if __name__ == '__main__':
    pass
    # Preprocessed data is saved in the data file
    # train_path = os.path.abspath('data/Dev.hdf5')
    # test_path = os.path.abspath('data/Test.hdf5')
    # x_train, y_train, x_test, y_test = train_test_set(train_path, test_path)
