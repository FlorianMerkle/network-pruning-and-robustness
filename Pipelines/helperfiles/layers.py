from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers




class ResBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels=3 ,output_channels = 64, stride=1, filter_size=3):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.conv1 = CustomConvLayer(
            (filter_size, filter_size, input_channels, output_channels),
            bias=False,
            stride=self.stride, 
        )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = CustomConvLayer(
            (filter_size, filter_size, input_channels, output_channels),
            bias=False,
            stride=1
        )
        self.bn2 = layers.BatchNormalization()
        if stride == 2:
            self.conv3 = CustomConvLayer(
                (filter_size, filter_size, input_channels, output_channels),
                bias=False,
                stride=self.stride, 
            )
            self.bn3 = layers.BatchNormalization()
        self.add1 = layers.Add()
    
    def call(self, inputs, training=False):
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        
        if self.stride == 2:
            inputs = self.conv3(x)
            inputs = self.bn3(x)
        return (self.add1([x, inputs]))

class CustomConvLayer(layers.Layer):

    def __init__(self, shape, bias=True, stride=1, padding='SAME'):

        super(CustomConvLayer, self).__init__()
        self.bias = bias
        self.w = self.add_weight(
            shape=shape,
            initializer='glorot_uniform',
            trainable=True,
            name='w'
        )
        self.m = self.add_weight(
            shape=shape,
            initializer='ones',
            trainable=False,
            name='m'
        )
        if self.bias==True:
            self.b = self.add_weight(
                shape=shape[-1],
                initializer='zeros',
                trainable=True,
                name='b'
            )
        self.s = stride
        self.p = padding
        
    def call(self, inputs):
        x = tf.nn.conv2d(inputs, tf.multiply(self.w, self.m), strides=[1, self.s, self.s, 1], padding=self.p,)
        if self.bias == True:
            x = tf.nn.bias_add(x, self.b)
        
        return tf.nn.relu(x)
    
#Dense Layer with Bias
class CustomDenseLayer(layers.Layer):
    def __init__(self, shape, bias, activation = 'relu'):
        super(CustomDenseLayer, self).__init__()
        self.bias = bias
        self.w = self.add_weight(
            shape = shape,
            initializer='glorot_uniform',
            trainable = True,
            name='w'
        )
        self.m = self.add_weight(
            shape = shape,
            initializer='ones',
            trainable = False,
            name='m'
        )
        if self.bias == True:
            self.b = self.add_weight(
                shape = (shape[-1]),
                initializer = 'zeros',
                trainable = True,
                name='b'
            )
        self.a = activation
        
        
    def call(self, inputs):
        x = tf.matmul(inputs, tf.multiply(self.w, self.m))
        if self.bias == True:
            x = tf.nn.bias_add(x, self.b)
        if self.a == 'relu':
            return tf.nn.tanh(x)
        if self.a == 'softmax':
            return tf.nn.softmax(x)
    
#Average Pooling Layer
class CustomPoolLayer(layers.Layer):
    
    def __init__(self, k=2, padding='SAME'):#padding='VALID'):
        super(CustomPoolLayer, self).__init__()
        self.k = k
        self.p = padding
    
    def call(self, inputs):
        return tf.nn.max_pool2d(inputs, ksize=[1, self.k, self.k,1], strides=[1, self.k, self.k, 1], padding=self.p)