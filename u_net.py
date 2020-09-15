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
    def __init__(self, kernel_size = (5,5), strides = (2,2)):
        # Initialize Model properties
        super(VocalUNet, self).__init__()
        '''
        VocalUNet is the linked paper above translated into Keras Model classes.

        Parameters:
        -----------
        - inputs: (array [might be tf.Input we will see.])
        Inputs the model takes in. Should be preprocessed MFCC data from preprocessing.py
        INPUTS MIGHT BE REMOVED.
        - kernel_size: (tuple/list)
        Size of kernel for model. Default set to (5,5).

        - strides: (tuple/list)
        Size of stride of kernel. Default set to (2,2).

        NEED TO ADD TO INIT AND DETERMINE HOW TO USE THIS PROPERLY.
        - training: (bool)
        True or false for differentiating between training/testing.
        '''
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
        '''
        Deconvolve/Convolution Transpose layers:
        As we deconvolve the layers, we dropout half for the first three
        deconvolution layers. The rest follows typical procedure of
        deconvoloution with sigmoid activation function.
        '''
        # Convolution Transpose 1:
        self.convt1 = BatchNormalization(Conv2DTranspose(inputs = self.conv6,
        filters = 256, kernel_size = kernel_size, strides = strides,
        activation = relu))
        # Dropout 1:
        self.dropout1 = Dropout(rate = 0.5, inputs = self.convt1,
        training = training)
        # Concatnate 1:
        self.con1 = concatenate([self.dropout1, self.conv5], 3)
        # Convolution Transpose 2:
        self.convt2 = BatchNormalization(Conv2DTranspose(inputs = self.con1,
        filters = 128, kernel_size = kernel_size, strides = strides,
        activation = relu))
        # Dropout 2:
        self.dropout2 = Dropout(rate = 0.5, inputs = self.convt1,
        training = training)
        # Concatenate 2:
        self.con2 = concatenate([self.dropout2, self.conv4],3)
        # Convolution Transpose 3:
        self.convt3 = BatchNormalization(Conv2DTranspose(inputs = self.con2,
        filters = 64, kernel_size = kernel_size, strides = strides,
        activation = relu))
        # Dropout 3:
        self.dropout3 = Dropout(rate = 0.5, inputs = self.convt1,
        training = training)
        # Concatenate 3:
        self.con3 = concatenate([self.dropout3, self.conv3],3)
        # Convolution Transpose 4:
        self.convt4 = BatchNormalization(Conv2DTranspose(inputs = self.con3,
        filters = 32, kernel_size = kernel_size, strides = strides,
        activation = relu))
        # Concatenate 4:
        self.con4 = concatenate([self.convt4, self.conv2],3)
        # Convolution Transpose 5:
        self.convt5 = BatchNormalization(Conv2DTranspose(inputs = self.con4,
        filters = 16, kernel_size = kernel_size, strides = strides,
        activation = relu))
        # Concatenate 5:
        self.con5 = concatenate([self.convt5, self.conv1],3)
        # Convolution Transpose 6:
        self.convt6 = BatchNormalization(Conv2DTranspose(inputs = self.conv6,
        filters = 1, kernel_size = kernel_size, strides = strides,
        activation = sigmoid))
    # Implementing model's forward pass:
    def call(self, inputs)

if __name__ == '__main__':
    pass
    # Preprocessed data is saved in the data file
    # train_path = os.path.abspath('data/Dev.hdf5')
    # test_path = os.path.abspath('data/Test.hdf5')
    # x_train, y_train, x_test, y_test = train_test_set(train_path, test_path)
