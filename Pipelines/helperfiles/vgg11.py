from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from tqdm import tqdm
import matplotlib.pyplot as plt
import pathlib
import os
import random
from secrets import randbelow
import foolbox as fb
from datetime import datetime

#import helperfiles.helpers as helpers

shapes = {

    'conv_1': (3, 3, 3, 64),
    'conv_2': (3, 3, 64, 128),
    'conv_3': (3, 3, 128, 256),
    'conv_4': (3, 3, 256, 256),
    'conv_5': (3, 3, 256, 512),
    'conv_6': (3, 3, 512, 512),
    'conv_7': (3, 3, 512, 512),
    'conv_8': (3, 3, 512, 512),
    'dense_1': (7*7*512, 4096),
    'dense_2': (4096, 1024),
    'dense_3': (1024, 10),
}

#conv2D with bias and relu activation

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
        

#Average Pooling Layer
class CustomPoolLayer(layers.Layer):
    
    def __init__(self, k=2, padding='SAME'):#padding='VALID'):
        super(CustomPoolLayer, self).__init__()
        self.k = k
        self.p = padding
    
    def call(self, inputs):
        return tf.nn.max_pool2d(inputs, ksize=[1, self.k, self.k,1], strides=[1, self.k, self.k, 1], padding=self.p)
    
#Dense Layer with Bias
class CustomDenseLayer(layers.Layer):
    def __init__(self, shape, bias, activation = 'relu'):
        super(CustomDenseLayer, self).__init__()
        self.bias = bias
        self.w = self.add_weight(
            shape = shape,
            initializer='random_normal',
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
        
class VGG11(tf.keras.Model):
    def __init__(self):
        super(VGG11, self).__init__()
        self.conv1 = CustomConvLayer(shapes['conv_1'], False, 1,)
        self.maxpool1 = CustomPoolLayer(k=2)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = CustomConvLayer(shapes['conv_2'], False, 1,)
        self.maxpool2 = CustomPoolLayer(k=2)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = CustomConvLayer(shapes['conv_3'], False, 1,)
        self.bn3 = layers.BatchNormalization()
        self.conv4 = CustomConvLayer(shapes['conv_4'], False, 1,)
        self.maxpool3 = CustomPoolLayer(k=2)
        self.bn4 = layers.BatchNormalization()
        self.conv5 = CustomConvLayer(shapes['conv_5'], False, 1,)
        self.bn5 = layers.BatchNormalization()
        self.conv6 = CustomConvLayer(shapes['conv_6'], False, 1,)
        self.maxpool4 = CustomPoolLayer(k=2)
        self.bn6 = layers.BatchNormalization()
        self.conv7 = CustomConvLayer(shapes['conv_7'], False, 1,)
        self.bn7 = layers.BatchNormalization()
        self.conv8 = CustomConvLayer(shapes['conv_8'], False, 1,)
        self.maxpool5 = CustomPoolLayer(k=2)
        self.bn8 = layers.BatchNormalization()
        self.dense1 = CustomDenseLayer(shapes['dense_1'], True, 'relu')
        #self.bn9 = layers.BatchNormalization()
        self.dense2 = CustomDenseLayer(shapes['dense_2'], True, 'relu')
        #self.bn10 = layers.BatchNormalization()
        self.dense3 = CustomDenseLayer(shapes['dense_3'], True, 'softmax')
        self.conv_layers = [0, 6, 12, 18, 24, 30, 36, 42]
        self.conv_masks = [1, 7, 13, 19, 25, 31, 37, 43]
        self.dense_layers = [48, 51, 54]
        self.dense_masks = [50, 53, 56]
        #self.conv_layers = []
        #self.conv_masks = []
        #self.dense_layers = []
        #self.dense_masks = []
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.maxpool3(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.maxpool4(x)
        x = self.bn6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.conv8(x)
        x = self.maxpool5(x)
        x = self.bn8(x)
        x = layers.Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
    
    
        