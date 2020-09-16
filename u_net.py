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
        - kernel_size: (tuple/list)
        Size of kernel for model. Default set to (5,5).

        - strides: (tuple/list)
        Size of stride of kernel. Default set to (2,2).

        NOTES:
        - need to add BatchNormalization/ Dropout layers
        - Since they are similar, I could initialize them once.
        - need to know what to do with training paramter for dropout.
        (might not need to worry about this.)
        '''
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
        # Convolution Transpose 2:
        self.convt2 = Conv2DTranspose(filters = 128,kernel_size = kernel_size,
        strides = strides, activation = relu)
        # Convolution Transpose 3:
        self.convt3 = Conv2DTranspose(filters = 64,kernel_size = kernel_size,
        strides = strides, activation = relu)
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
        pass
        # p represents pass. p1 therefore is pass 1, p2 is pass 2, and so on.
        # p1 = self.conv1(inputs)
        # p2 = self.bn1(p1)
        # p3 =

if __name__ == '__main__':
    pass
    # Preprocessed data is saved in the data file
    # train_path = os.path.abspath('data/Dev.hdf5')
    # test_path = os.path.abspath('data/Test.hdf5')
    # x_train, y_train, x_test, y_test = train_test_set(train_path, test_path)
