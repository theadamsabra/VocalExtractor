# Clean this up later
import tensorflow as tf
import numpy as np
from tensorflow.nn import leaky_relu, relu
from tensorflow.keras.layers import BatchNormalization, Dropout, Conv2D, Conv2DTranspose

'''
The loss function used to train the model is the L 1
norm of the difference of the target spectrogram and the
masked input spectrogram
'''
class VocalUNetLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return np.linalg.norm(y_pred - y_true, ord=1)
class VocalUNet(tf.keras.Model):
    def __init__(self, kernel_size = (5,5), strides = (2,2)):
        # Initialize Model properties
        super(VocalUNet, self).__init__()
        '''
        VocalUNet is the linked paper above translated into Keras Model classes.

        Parameters:
        -----------
        - kernel_size: (tuple/list)
        Size of kernel for model. Default set to (5,5).

        - strides: (tuple/list)
        Size of stride of kernel. Default set to (2,2).
        '''
        # First Convolution
        self.conv1 = Conv2D(filters = 16, kernel_size = kernel_size,
        strides = strides, activation = leaky_relu)
        # First Batch Norm
        self.bn1 = BatchNormalization()
        # Second Convolution
        self.conv2 = Conv2D(filters = 32, kernel_size = kernel_size,
        strides = strides, activation = leaky_relu)
        # Second Batch Norm
        self.bn2 = BatchNormalization()
        # Third Convolution
        self.conv3 = Conv2D(filters = 64, kernel_size = kernel_size,
        strides = strides, activation = leaky_relu)
        # Third Batch Norm
        self.bn3 = BatchNormalization()
        # Fourth Convolution
        self.conv4 = Conv2D(filters = 128, kernel_size = kernel_size,
        strides = strides, activation = leaky_relu)
        # Fourth Batch Norm
        self.bn4 = BatchNormalization()
        # Fifth Convolotion
        self.conv5 = Conv2D(filters = 256, kernel_size = kernel_size,
        strides = strides, activation = leaky_relu)
        # Fifth Batch Norm
        self.bn5 = BatchNormalization()
        # Sixth Convolution
        self.conv6 = Conv2D(filters = 512, kernel_size = kernel_size,
        strides = strides, activation = leaky_relu)
        # Sixth Batch Norm
        self.bn6 = BatchNormalization()
        '''
        Deconvolve/Convolution Transpose layers:
        As we deconvolve the layers, we dropout half for the first three
        deconvolution layers. The rest follows typical procedure of
        deconvoloution with sigmoid activation function.
        '''
        # Convolution Transpose 1:
        self.convt1 = Conv2DTranspose(filters = 256, kernel_size = kernel_size,
        strides = strides, activation = relu)
        # Seventh Batch Norm:
        self.bn7 = BatchNormalization()
        # First Dropout:
        self.dropout1 = Dropout(0.5)
        # Convolution Transpose 2:
        self.convt2 = Conv2DTranspose(filters = 128,kernel_size = kernel_size,
        strides = strides, activation = relu)
        # Eight Batch Norm:
        self.bn8 = BatchNormalization()
        # Second Dropout:
        self.dropout2 = Dropout(0.5)
        # Convolution Transpose 3:
        self.convt3 = Conv2DTranspose(filters = 64,kernel_size = kernel_size,
        strides = strides, activation = relu)
        # Ninth Batch Norm:
        self.bn9 = BatchNormalization()
        # Third Dropout:
        self.dropout3 = Dropout(0.5)
        # Convolution Transpose 4:
        self.convt4 = Conv2DTranspose(filters = 32,kernel_size = kernel_size,
        strides = strides, activation = relu)
        # Tenth Batch Norm
        self.bn10 = BatchNormalization()
        # Convolution Transpose 5:
        self.convt5 = Conv2DTranspose(filters = 16,kernel_size = kernel_size,
        strides = strides, activation = relu)
        # Eleventh Batch Norm
        self.bn11 = BatchNormalization()
        # Convolution Transpose 6:
        self.convt6 = Conv2DTranspose(filters = 1, kernel_size = kernel_size,
        strides = strides, activation = sigmoid)
    def call(self, inputs):
        '''
        Again, typical syntax on TensorFlow would prefer x as the variable
        name. However, if I keep this syntax for my use case, concatenating
        specific layers will be a mess. To avoid this, I will name each layer
        in the forward pass (i.e. l1, l2, etc.)
        '''
        l1 = self.bn1(self.conv1(inputs))
        l2 = self.bn2(self.conv2(l1))
        l3 = self.bn3(self.conv3(l2))
        l4 = self.bn4(self.conv4(l3))
        l5 = self.bn5(self.conv5(l4))
        l6 = self.bn6(self.conv6(l5))
        l7 = self.bn7(self.convt1(l6))
        l8 = self.dropout1(l7)
        l9 = self.bn8(self.convt2(concatenate([l8, l5])))
        l10 = self.dropout2(l9)
        l11 = self.bn9(self.convt3(concatenate([l10, l4])))
        l12 = self.dropout3(l11)
        l13 = self.bn10(self.convt4(concatenate([l12, l3])))
        l14 = self.bn11(self.convt5(concatenate([l13, l2])))
        l15 = self.convt6(concatenate([l14,l1]))
        return l15 * inputs
