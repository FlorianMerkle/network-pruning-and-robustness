#todo create custom model class subclassing tf.keras.Model with generic pruning methods

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


#from . import helpers
from .pruning import _prune_random_local_unstruct, _prune_magnitude_global_unstruct, _prune_random_local_struct, _prune_magnitude_local_struct, _prune_magnitude_global_struct, _prune_magnitude_local_unstruct

from .layers import CustomConvLayer, CustomDenseLayer, ResBlock, CustomPoolLayer

from .helpers import _find_layers_and_masks


class ResNet8(tf.keras.Model):
    def __init__(self):
        super(ResNet8, self).__init__()
        self.conv1 = CustomConvLayer(
            (3,3,3,32),
            bias=False,
            stride=2
        )
        self.res_block1 = ResBlock(32, 64, 2)
        self.res_block2 = ResBlock(64, 128, 2)
        self.res_block3 = ResBlock(128, 256, 2)
        self.pool2 = layers.GlobalAveragePooling2D()
        self.dense1 = CustomDenseLayer((256, 10), True, activation='softmax')
        self.conv_layers = []
        self.conv_masks = []
        self.dense_layers = []
        self.dense_masks = []
        #self.find_layers_and_masks()

    def call(self,inputs, training=False):

        x = self.conv1(inputs)
        #x = self.pool1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.pool2(x)
        x = self.dense1(x)
        return x
    
    def prune_random_local_unstruct(self, ratio):
        self.find_layers_and_masks()
        weights = self.get_weights()
        weights = _prune_random_local_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def prune_magnitude_global_unstruct(self, ratio):
        self.find_layers_and_masks()
        weights = self.get_weights()
        weights = _prune_magnitude_global_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def prune_random_local_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        self.find_layers_and_masks()
        weights = self.get_weights()
        weights = _prune_random_local_struct(self, ratio, weights, prune_dense_layers=prune_dense_layers, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_random_global_struct(self, ratio):
        raise Warning('Not yet implemented')
        return False
    def prune_magnitude_local_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        self.find_layers_and_masks()
        weights = self.get_weights()
        weights = _prune_magnitude_local_struct(self, ratio, weights, prune_dense_layers=prune_dense_layers, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_magnitude_global_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        self.find_layers_and_masks()
        weights = self.get_weights()
        weights = _prune_magnitude_global_struct(self, ratio, weights, prune_dense_layers=False, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_magnitude_local_unstruct(self, ratio):
        self.find_layers_and_masks()
        weights = self.get_weights()
        weights = _prune_magnitude_local_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def find_layers_and_masks(self):
        _find_layers_and_masks(self)
        return True
    
    

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = CustomConvLayer(
            (7,7,3,64),
            bias=False,
            stride=2
        )
        self.pool1 = layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')
        self.res_block1 = ResBlock(64, 64)
        #self.res_block2 = ResBlock(64, 64)
        self.res_block3 = ResBlock(64, 64)
        self.res_block4 = ResBlock(64, 128, 2)
        #self.res_block5 = ResBlock(128, 128)
        #self.res_block6 = ResBlock(128, 128)
        self.res_block7 = ResBlock(128, 128)
        self.res_block8 = ResBlock(128, 256, 2)
        #self.res_block9 = ResBlock(256, 256)
        #self.res_block10 = ResBlock(256, 256)
        #self.res_block11 = ResBlock(256, 256)
        #self.res_block12 = ResBlock(256, 256)
        self.res_block13 = ResBlock(256, 256)
        self.res_block14 = ResBlock(256 ,512, 2)
        #self.res_block15 = ResBlock(512, 512)
        self.res_block16 = ResBlock(512, 512)
        self.pool2 = layers.GlobalAveragePooling2D()
        self.dense1 = CustomDenseLayer((512, 10), True, activation='softmax')
        #self.conv_layers = []
        #self.conv_masks = []
        #self.dense_layers = []
        #self.dense_masks = []
        self.conv_layers = [0, 2, 5, 14, 17, 26, 29, 40, 43, 52, 55, 66, 69, 78, 81, 92, 95]
        self.conv_masks = [1, 8, 11, 20, 23, 33, 36, 46, 49, 59, 62, 72, 75, 85, 88, 98, 101]
        self.dense_layers = [104]
        self.dense_masks = [106]
        


        
        
    def call(self,inputs, training=False):

        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.res_block1(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block7(x)
        x = self.res_block8(x)
        x = self.res_block13(x)
        x = self.res_block14(x)
        x = self.res_block16(x)
        x = self.pool2(x)
        x = self.dense1(x)
        return x
    
    def prune_random_local_unstruct(self, ratio):
        weights = self.get_weights()
        weights = _prune_random_local_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def prune_magnitude_global_unstruct(self, ratio):
        weights = self.get_weights()
        weights = _prune_magnitude_global_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def prune_random_local_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        weights = self.get_weights()
        weights = _prune_random_local_struct(self, ratio, weights, prune_dense_layers=prune_dense_layers, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_random_global_struct(self, ratio):
        raise Warning('Not yet implemented')
        return False
    def prune_magnitude_local_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        weights = self.get_weights()
        weights = _prune_magnitude_local_struct(self, ratio, weights, prune_dense_layers=prune_dense_layers, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_magnitude_global_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        weights = self.get_weights()
        weights = _prune_magnitude_global_struct(self, ratio, weights, prune_dense_layers=False, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_magnitude_local_unstruct(self, ratio):
        weights = self.get_weights()
        weights = _prune_magnitude_local_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def find_layers_and_masks(self):
        _find_layers_and_masks(self)
        return True
    
    
    
vgg_shapes = {

    'conv_1': (3, 3, 3, 64),
    'conv_2': (3, 3, 64, 128),
    'conv_3': (3, 3, 128, 256),
    'conv_4': (3, 3, 256, 256),
    'conv_5': (3, 3, 256, 512),
    'conv_6': (3, 3, 512, 512),
    'conv_7': (3, 3, 512, 512),
    'conv_8': (3, 3, 512, 512),
    #'dense_1': (7*7*512, 4096),
    'dense_2': (2048, 1024),
    'dense_3': (1024, 10),
}
        
class VGG11(tf.keras.Model):
    def __init__(self):
        super(VGG11, self).__init__()
        self.conv1 = CustomConvLayer(vgg_shapes['conv_1'], False, 1,)
        #self.maxpool1 = CustomPoolLayer(k=2)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = CustomConvLayer(vgg_shapes['conv_2'], False, 1,)
        self.maxpool2 = CustomPoolLayer(k=2)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = CustomConvLayer(vgg_shapes['conv_3'], False, 1,)
        self.bn3 = layers.BatchNormalization()
        self.conv4 = CustomConvLayer(vgg_shapes['conv_4'], False, 1,)
        self.maxpool3 = CustomPoolLayer(k=2)
        self.bn4 = layers.BatchNormalization()
        self.conv5 = CustomConvLayer(vgg_shapes['conv_5'], False, 1,)
        self.bn5 = layers.BatchNormalization()
        self.conv6 = CustomConvLayer(vgg_shapes['conv_6'], False, 1,)
        self.maxpool4 = CustomPoolLayer(k=2)
        self.bn6 = layers.BatchNormalization()
        self.conv7 = CustomConvLayer(vgg_shapes['conv_7'], False, 1,)
        self.bn7 = layers.BatchNormalization()
        self.conv8 = CustomConvLayer(vgg_shapes['conv_8'], False, 1,)
        self.maxpool5 = CustomPoolLayer(k=2)
        self.bn8 = layers.BatchNormalization()
        #self.dense1 = CustomDenseLayer(vgg_shapes['dense_1'], True, 'relu')
        #self.bn9 = layers.BatchNormalization()
        self.dense2 = CustomDenseLayer(vgg_shapes['dense_2'], True, 'relu')
        #self.bn10 = layers.BatchNormalization()
        self.dense3 = CustomDenseLayer(vgg_shapes['dense_3'], True, 'softmax')
        self.conv_layers = [0, 6, 12, 18, 24, 30, 36, 42]
        self.conv_masks = [1, 7, 13, 19, 25, 31, 37, 43]
        self.dense_layers = [48, 51]
        self.dense_masks = [50, 53]
        #self.conv_layers = []
        #self.conv_masks = []
        #self.dense_layers = []
        #self.dense_masks = []
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        #x = self.maxpool1(x)
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
        #x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
    def prune_random_local_unstruct(self, ratio):
        _find_layers_and_masks(self)
        weights = self.get_weights()
        weights = _prune_random_local_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def prune_magnitude_global_unstruct(self, ratio):
        _find_layers_and_masks(self)
        weights = self.get_weights()
        weights = _prune_magnitude_global_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def prune_random_local_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        _find_layers_and_masks(self)
        weights = self.get_weights()
        weights = _prune_random_local_struct(self, ratio, weights, prune_dense_layers=prune_dense_layers, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_random_global_struct(self, ratio):
        raise Warning('Not yet implemented')
        return False
    def prune_magnitude_local_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        _find_layers_and_masks(self)
        weights = self.get_weights()
        weights = _prune_magnitude_local_struct(self, ratio, weights, prune_dense_layers=prune_dense_layers, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_magnitude_global_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        _find_layers_and_masks(self)
        weights = self.get_weights()
        weights = _prune_magnitude_global_struct(self, ratio, weights, prune_dense_layers=False, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_magnitude_local_unstruct(self, ratio):
        _find_layers_and_masks(self)
        weights = self.get_weights()
        weights = _prune_magnitude_local_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def find_layers_and_masks(self):
        _find_layers_and_masks(self)
        return False
    
cnn_shapes = {
    # 5x5 conv, 1 input, 6 outputs
    'conv_1': (5, 5, 1, 6),
    # 5x5 conv, 6 inputs, 16 outputs
    'conv_2': (5, 5, 6, 16),
    #5x5 conv as in paper, 16 inputs, 120 outputs
    'conv_3': (1, 1, 16, 120),
    # fully connected, 5*5*16 inputs, 120 outputs
    'dense_1': (5*5*16, 120),
    # fully connected, 120 inputs, 84 outputs
    'dense_2': (120, 84),
    # 84 inputs, 10 outputs (class prediction)
    'dense_3': (84, 10),
}
    
class CNN(tf.keras.Model):
    def model(self):
        x = Input(shape=(28*28))
        return Model(inputs=[x], outputs=self.call(x))

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = CustomConvLayer(cnn_shapes['conv_1'], True, 1, 'SAME')#'VALID')
        self.maxpool1 = CustomPoolLayer(k=2, padding='SAME')
        self.conv2 = CustomConvLayer(cnn_shapes['conv_2'], True, 1, 'VALID')
        self.maxpool2 = CustomPoolLayer(k=2, padding='VALID')

        self.dense1 = CustomDenseLayer(cnn_shapes['dense_1'], True, 'relu')
        self.dense2 = CustomDenseLayer(cnn_shapes['dense_2'], True, 'relu')
        self.dense3 = CustomDenseLayer(cnn_shapes['dense_3'], True, 'softmax')
        #self.conv_layers = []
        #self.conv_masks = []
        #self.dense_layers = []
        #self.dense_masks = []
        self.conv_layers = [0, 3]
        self.conv_masks = [2, 5]
        self.dense_layers = [6, 9, 12]
        self.dense_masks = [8, 11, 14]
        
        
    def call(self, inputs):
        x = tf.reshape(inputs, shape=[-1,28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = layers.Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
    def prune_random_local_unstruct(self, ratio):
        weights = self.get_weights()
        weights = _prune_random_local_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def prune_magnitude_global_unstruct(self, ratio):
        weights = self.get_weights()
        weights = _prune_magnitude_global_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def prune_random_local_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        weights = self.get_weights()
        weights = _prune_random_local_struct(self, ratio, weights, prune_dense_layers=prune_dense_layers, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_random_global_struct(self, ratio):
        raise Warning('Not yet implemented')
        return False
    def prune_magnitude_local_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        weights = self.get_weights()
        weights = _prune_magnitude_local_struct(self, ratio, weights, prune_dense_layers=prune_dense_layers, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_magnitude_global_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        weights = self.get_weights()
        weights = _prune_magnitude_global_struct(self, ratio, weights, prune_dense_layers=False, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_magnitude_local_unstruct(self, ratio):
        weights = self.get_weights()
        weights = _prune_magnitude_local_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def find_layers_and_masks(self):
        return False
    
    
class LeNet300_100(tf.keras.Model):
    def __init__(self):
        super(LeNet300_100, self).__init__()
        self.dense1 = CustomDenseLayer((28*28,300), True)
        self.dense2 = CustomDenseLayer((300,100), True)
        self.dense3 = CustomDenseLayer((100,10), True, activation='softmax')
        self.conv_layers = []
        self.conv_masks = []
        self.dense_layers = [0,3,6]
        self.dense_masks = [2,5,8]
        
    def call(self, inputs):
        x = layers.Flatten()(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)
    
    def prune_random_local_unstruct(self, ratio):
        weights = self.get_weights()
        weights = _prune_random_local_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def prune_magnitude_global_unstruct(self, ratio):
        weights = self.get_weights()
        weights = _prune_magnitude_global_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def prune_random_local_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        weights = self.get_weights()
        weights = _prune_random_local_struct(self, ratio, weights, prune_dense_layers=prune_dense_layers, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_random_global_struct(self, ratio):
        raise Warning('Not yet implemented')
        return False
    def prune_magnitude_local_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        weights = self.get_weights()
        weights = _prune_magnitude_local_struct(self, ratio, weights, prune_dense_layers=prune_dense_layers, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_magnitude_global_struct(self, ratio, prune_dense_layers=False, structure_to_prune='filter'):
        weights = self.get_weights()
        weights = _prune_magnitude_global_struct(self, ratio, weights, prune_dense_layers=False, structure=structure_to_prune)
        self.set_weights(weights)
        return True
    def prune_magnitude_local_unstruct(self, ratio):
        weights = self.get_weights()
        weights = _prune_magnitude_local_unstruct(self, ratio, weights)
        self.set_weights(weights)
        return True
    def find_layers_and_masks(self):
        return False
    
    