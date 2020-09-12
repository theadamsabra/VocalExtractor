# Clean this up later
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.nn import leaky_relu, relu
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

# Custom loss function for VocalUNet
class u_net_loss(tf.keras.losses.Loss):
    pass

class VocalUNet(tf.keras.Model):
    def __init__(self, inputs, kernel_size, strides):
        super(VocalUNet, self).__init__() # Initialize Model properties
        # Initialize network - Convolve down to 512 filters.
        # First Convolution
        self.conv1 = BatchNormalization(Conv2D(inputs = inputs,
        filters = 16, kernel_size = kernel_size, strides = strides,
        activation = leaky_relu))
        # Second Convolution
        self.conv2 = BatchNormalization(Conv2D(inputs = self.conv1,
        filters = 32, kernel_size = kernel_size, strides = strides,
        activation = leaky_relu))
        # Third Convolution
        self.conv3 = BatchNormalization(Conv2D(inputs = self.conv2,
        filters = 64, kernel_size = kernel_size, strides = strides,
        activation = leaky_relu))
        # Fourth Convolution
        self.conv4 = BatchNormalization(Conv2D(inputs = self.conv3,
        filters = 128, kernel_size = kernel_size, strides = strides,
        activation = leaky_relu))
        # Fifth Convolotion
        self.conv5 = BatchNormalization(Conv2D(inputs = self.conv4,
        filters = 256, kernel_size = kernel_size, strides = strides,
        activation = leaky_relu))
        # Sixth Convolution
        self.conv6 = BatchNormalization(Conv2D(inputs = self.conv5,
        filters = 512, kernel_size = kernel_size, strides = strides,
        activation = leaky_relu))
        # Deconvolve/Convolution Transpose layers.
        '''
        - Link inputs together
        - add dropouts
        '''
        # Convolution Transpose 1:
        self.convt1 = BatchNormalization(Conv2DTranspose(inputs = self.conv6,
        filters = 256, kernel_size = kernel_size, strides = strides,
        activation = relu))
        # Convolution Transpose 2:
        self.convt2 = BatchNormalization(Conv2DTranspose(inputs = self.conv6,
        filters = 128, kernel_size = kernel_size, strides = strides,
        activation = relu))
        # Convolution Transpose 3:
        self.convt3 = BatchNormalization(Conv2DTranspose(inputs = self.conv6,
        filters = 64, kernel_size = kernel_size, strides = strides,
        activation = relu))
        # Convolution Transpose 4:
        self.convt4 = BatchNormalization(Conv2DTranspose(inputs = self.conv6,
        filters = 32, kernel_size = kernel_size, strides = strides,
        activation = relu))
        # Convolution Transpose 5:
        self.convt5 = BatchNormalization(Conv2DTranspose(inputs = self.conv6,
        filters = 16, kernel_size = kernel_size, strides = strides,
        activation = relu))
        # Convolution Transpose 6:
        self.convt6 = BatchNormalization(Conv2DTranspose(inputs = self.conv6,
        filters = 1, kernel_size = kernel_size, strides = strides,
        activation = relu))
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
