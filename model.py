import h5py
import numpy as np
import os
# Clean this up later
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.nn import leaky_relu, relu
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

# Functions for bringing in the test/train data - might be imported as
# separate file.
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
class u_net_loss(tf.keras.losses.Loss):
    pass

class VocalUNet(tf.keras.Model):
    def __init__(self, inputs, kernel_size, strides):
        super(VocalUNet, self).__init__()
        # Initialize parameters
        self.inputs = inputs
        self.kernel_size = kernel_size
        self.strides = strides
        # Initialize network - Convolve down to 512 filters.
        self.conv1 = BatchNormalization(Conv2D(inputs = self.inputs,
        filters = 16, kernel_size = self.kernel_size, strides = self.strides,
        activation = leaky_relu))
        self.conv2 = BatchNormalization(Conv2D(inputs = self.inputs,
        filters = 32, kernel_size = self.kernel_size, strides = self.strides,
        activation = leaky_relu))
        self.conv3 = BatchNormalization(Conv2D(inputs = self.inputs,
        filters = 64, kernel_size = self.kernel_size, strides = self.strides,
        activation = leaky_relu))
        self.conv4 = BatchNormalization(Conv2D(inputs = self.inputs,
        filters = 128, kernel_size = self.kernel_size, strides = self.strides,
        activation = leaky_relu))
        self.conv5 = BatchNormalization(Conv2D(inputs = self.inputs,
        filters = 256, kernel_size = self.kernel_size, strides = self.strides,
        activation = leaky_relu))
        self.conv6 = BatchNormalization(Conv2D(inputs = self.inputs,
        filters = 512, kernel_size = self.kernel_size, strides = self.strides,
        activation = leaky_relu))
        # Deconvolve layers.
        pass
    # Function to train the model
    def train_model(self, x_train, y_train):
        # if x_train.shape == y_train.shape:
        #     self.rows = x_train.shape[1]      # Maybe.
        #     self.cols = x_train.shape[2]
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
