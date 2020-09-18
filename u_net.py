# Clean this up later
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.nn import leaky_relu, relu
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

# Loss function for VocalUNet
class VocalUNetLoss(tf.keras.losses.Loss):
    def call(self, y, y_hat):
        pass

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

        A Note on Batch Normalization and Dropout layers:
        ------------------------------------------------
        Batch Normalization and Dropouts are used throughout the
        convolution and convolution transpose. Syntax among the
        TensorFlow Model Classes dictates that these layers - even
        if the same - are initalized numerous times. However, in
        the Vocal U-Net, these layers are the same. I will initalize
        them once and reuse them throughout the call() function.

        NOTES:
        - need to know what to do with training paramter for dropout.
        '''
        # Batch Normalization layer:
        self.bn = BatchNormalization()
        # Dropout layer:
        self.dropout = Dropout(0.5)
        # First Convolution
        self.conv1 = Conv2D(filters = 16, kernel_size = kernel_size,
        strides = strides, activation = leaky_relu)
        # Second Convolution
        self.conv2 = Conv2D(filters = 32, kernel_size = kernel_size,
        strides = strides, activation = leaky_relu)
        # Third Convolution
        self.conv3 = Conv2D(filters = 64, kernel_size = kernel_size,
        strides = strides, activation = leaky_relu)
        # Fourth Convolution
        self.conv4 = Conv2D(filters = 128, kernel_size = kernel_size,
        strides = strides, activation = leaky_relu)
        # Fifth Convolotion
        self.conv5 = Conv2D(filters = 256, kernel_size = kernel_size,
        strides = strides, activation = leaky_relu)
        # Sixth Convolution
        self.conv6 = Conv2D(filters = 512, kernel_size = kernel_size,
        strides = strides, activation = leaky_relu)
        '''
        Deconvolve/Convolution Transpose layers:
        As we deconvolve the layers, we dropout half for the first three
        deconvolution layers. The rest follows typical procedure of
        deconvoloution with sigmoid activation function.
        '''
        # Convolution Transpose 1:
        self.convt1 = Conv2DTranspose(filters = 256, kernel_size = kernel_size,
        strides = strides, activation = relu)
        # Dropout 1:
        self.dropout1 = Dropout(rate = 0.5)
        # Convolution Transpose 2:
        self.convt2 = Conv2DTranspose(filters = 128,kernel_size = kernel_size,
        strides = strides, activation = relu)
        # Dropout 2:
        self.dropout2 = Dropout(rate = 0.5)
        # Convolution Transpose 3:
        self.convt3 = Conv2DTranspose(filters = 64,kernel_size = kernel_size,
        strides = strides, activation = relu)
        # Dropout 3:
        self.dropout3 = Dropout(rate = 0.5)
        # Convolution Transpose 4:
        self.convt4 = Conv2DTranspose(filters = 32,kernel_size = kernel_size,
        strides = strides, activation = relu)
        # Convolution Transpose 5:
        self.convt5 = Conv2DTranspose(filters = 16,kernel_size = kernel_size,
        strides = strides, activation = relu)
        # Convolution Transpose 6:
        self.convt6 = Conv2DTranspose(filters = 1, kernel_size = kernel_size,
        strides = strides, activation = sigmoid)
    # Implementing model's forward pass:
    def call(self, inputs):
        '''
        Again, typical syntax on TensorFlow would prefer x as the variable
        name. However, if I keep this syntax below, concatenating specific
        layers will be a mess. To avoid this, I will name each layer in the
        forward pass as l1, l2, etc.
        '''
        l1 = self.bn(self.conv1(inputs))
        l2 = self.bn(self.conv2(l1))
        l3 = self.bn(self.conv3(l2))
        l4 = self.bn(self.conv4(l3))
        l5 = self.bn(self.conv5(l4))
        l6 = self.bn(self.conv6(l5))
        l7 = self.bn(self.convt1(l6))
        l8 = self.dropout(l7)
        l9 = self.bn(self.convt2(concatenate([l8, l5])))
        l10 = self.dropout(l9)
        l11 = self.bn(self.convt3(concatenate([l10, l4])))
        l12 = self.dropout(l11)
        l13 = self.bn(self.convt4(concatenate([l12, l3])))
        l14 = self.bn(self.convt5(concatenate([l13, l2])))
        l15 = self.convt6(concatenate([l14,l1]))
