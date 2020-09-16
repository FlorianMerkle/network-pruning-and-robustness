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

import helperfiles.helpers as helpers

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

    def __init__(self, shape, bias, stride, padding='SAME'):
        
        #super(CustomConvLayer, self).__init__()
        #self.w = weights
        #self.m = mask
        #self.b = biases
        #self.s = strides
        #self.p = padding
        #self.bn = layers.BatchNormalization()
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
        #x = self.bn(x)
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
    def prune_random_local_unstruct(self, ratio):
        def prune_conv_layers_locally(self, ratio):
            weights = self.get_weights()
            for i, layer in enumerate(self.conv_layers):
                #shape = 3,3,64,128
                converted_weights = helpers.convert_from_hwio_to_iohw(weights[layer]).numpy()
                converted_mask = helpers.convert_from_hwio_to_iohw(weights[self.conv_masks[i]]).numpy()
                #shape = 128,64, 3,3
                layer_shape = weights[layer].shape
                flat_masks = converted_mask.flatten()
                no_of_weighs_to_prune = int(np.round(ratio * len(flat_weights)))
                non_zero_weights = np.nonzero(flat_weights)[0]
                no_of_weights_to_prune_left = int(no_of_weighs_to_prune - (len(flat_weights) - len(non_zero_weights)) )
                random.shuffle(non_zero_weights)
                indices_to_delete = non_zero_weights[:no_of_weights_to_prune_left]
                for idx_to_delete in indices_to_delete:
                    flat_masks[idx_to_delete] = 0
                    flat_weights[idx_to_delete] = 0
                converted_mask = flat_masks.reshape(layer_shape)
                converted_weights = flat_weights.reshape(layer_shape)
                back_converted_mask = helpers.convert_from_iohw_to_hwio(converted_mask)
                back_converted_weights = helpers.convert_from_iohw_to_hwio(converted_weights)
                weights[layer] = back_converted_weights
                weights[self.conv_masks[i]] = back_converted_mask
            self.set_weights(weights)
            return True
            
        
        def prune_dense_layers_locally(self, ratio):
            weights = self.get_weights()
#            for index, weight in enumerate(weights):
            for i, layer in enumerate(self.dense_layers):
#                if index in dense_layer_to_prune:
                    shape = weights[layer].shape
                    flat_weights = weights[layer].flatten()
                    flat_mask = weights[self.dense_masks[i]].flatten()
                    no_of_weighs_to_prune = int(np.round(ratio * len(flat_weights)))
                    # find unpruned weights
                    non_zero_weights = np.nonzero(flat_mask)[0]
                    # calculate the amount of weights to be pruned this round
                    no_of_weights_to_prune_left = int(no_of_weighs_to_prune - (len(flat_weights) - len(non_zero_weights)) )
                    # shuffle all non-zero weights
                    random.shuffle(non_zero_weights)
                    # and take the indices of the first x weights where x is the number of weights to be pruned this round
                    indices_to_delete = non_zero_weights[:no_of_weights_to_prune_left]
                    for idx_to_delete in indices_to_delete:
                        flat_mask[idx_to_delete] = 0
                        flat_weights[idx_to_delete] = 0

                    mask_reshaped = flat_mask.reshape(shape)
                    weights_reshaped = flat_weights.reshape(shape)
                    weights[self.dense_masks[i]] = mask_reshaped
                    weights[layer] = weights_reshaped
            self.set_weights(weights)
            return weights
        weights = prune_conv_layers_locally(self, ratio)
        weights = prune_dense_layers_locally(self,ratio)
        return True
    
    def prune_magnitude_global_unstruct(self, ratio):

        weights = self.get_weights()
        flat_weights = []
        flat_mask = []
        all_masks = self.conv_masks + self.dense_masks
        for i, x in enumerate(self.conv_layers + self.dense_layers):
            flat_weights = np.append(flat_weights, weights[x].flatten())
            flat_mask = np.append(flat_mask, weights[all_masks[i]].flatten())
            
        no_of_weights_to_prune = int(np.round(len(flat_weights)*ratio))
        #print('total weights',len(flat_weights))
        #print('weights to prune w/o round',int(len(flat_weights)*ratio))
        #print('weights to prune with round',int(np.round(len(flat_weights)*ratio)))
        indices_to_delete = np.abs(flat_weights).argsort(0)[:no_of_weights_to_prune]
        
        for idx_to_delete in indices_to_delete:
            flat_mask[idx_to_delete] = 0
            flat_weights[idx_to_delete] = 0
        z = 0
        for i, x in enumerate(self.conv_layers + self.dense_layers):
            weights[x] = flat_weights[z:z + np.prod(weights[x].shape)].reshape(weights[x].shape)
            weights[all_masks[i]] = flat_mask[z:z + np.prod(weights[x].shape)].reshape(weights[x].shape)
            z = z + np.prod(weights[x].shape)            
        self.set_weights(weights)
        return True
    
    
    def prune_random_local_struct(self, ratio, prune_dense_layers=False):
        def prune_conv_layers(self, ratio):
            weights = self.get_weights()
            for i, layer in enumerate(self.conv_layers):

                vals = []
                iohw_weights = helpers.convert_from_hwio_to_iohw(weights[layer])
                iohw_mask = helpers.convert_from_hwio_to_iohw(weights[self.conv_masks[i]])
                converted_shape = iohw_weights.shape
                no_of_channels = converted_shape[0]*converted_shape[1]
                no_of_channels_to_prune = int(np.round(ratio * no_of_channels))
                channels = tf.reshape(iohw_weights, (no_of_channels,converted_shape[2],converted_shape[3])).numpy()
                #print(channels)
                non_zero_channels = np.nonzero([np.sum(channel) for channel in channels])[0]
                #print(non_zero_channels)
                no_of_channels_to_prune_left = no_of_channels_to_prune - (len(channels) - len(non_zero_channels))
                random.shuffle(non_zero_channels)
                channels_to_prune = non_zero_channels[:no_of_channels_to_prune_left]
                mask = tf.reshape(iohw_mask, 
                                  (no_of_channels,converted_shape[2],converted_shape[3])).numpy()

                for channel_to_prune in channels_to_prune:
                    channels[channel_to_prune] = tf.zeros([converted_shape[2],converted_shape[3]])
                    mask[channel_to_prune] = tf.zeros([converted_shape[2],converted_shape[3]])

                reshaped_mask = tf.reshape(mask, converted_shape)
                reshaped_weights = tf.reshape(channels, converted_shape)
                weights[layer] = helpers.convert_from_iohw_to_hwio(reshaped_weights)
                weights[self.conv_masks[i]] = helpers.convert_from_iohw_to_hwio(reshaped_mask)
            self.set_weights(weights)
            return True
        def prune_dense_layers(self, ratio):
            weights = self.get_weights()
            for i, layer_to_prune in enumerate(self.dense_layers):
                rows = weights[layer_to_prune]
                no_of_rows_to_prune = int(np.round(ratio * len(weights[layer_to_prune])))
                non_zero_rows = np.nonzero([np.sum(row) for row in rows])[0]
                no_of_rows_to_prune_left = no_of_rows_to_prune - (len(rows) - len(non_zero_rows))
                random.shuffle(non_zero_rows)
                rows_to_prune = non_zero_rows[:no_of_rows_to_prune_left]
                
                for row_to_prune in rows_to_prune:
                    weights[layer_to_prune][row_to_prune] = tf.zeros(len(weights[layer_to_prune][row_to_prune]))
                    weights[self.dense_masks[i]][row_to_prune] = tf.zeros(len(weights[layer_to_prune][row_to_prune]))
            self.set_weights(weights)
            return True
        prune_conv_layers(self, ratio)
        if prune_dense_layers==True:
            prune_dense_layers(self, ratio)
        
        return True

    def prune_random_global_struct(self, ratio, prune_dense_layers=False):
        raise Warning('Not yet implemented')
        return False
    
    def prune_magnitude_local_struct(self, ratio, structure='channel'):
        def prune_filters(self, ratio):
            weights = self.get_weights()
            for i, x in enumerate(self.conv_layers):
                # shape = (3,3,64,128)
                vals = []
                oihw_weights = helpers.convert_from_hwio_to_oihw(weights[x])
                oihw_mask = helpers.convert_from_hwio_to_oihw(weights[self.conv_masks[i]])
                # shape = (128,64,3,3)
                converted_shape = oihw_weights.shape
                no_of_filters = converted_shape[0]
                no_of_filters_to_prune = int(np.round(ratio * no_of_channels))
                for single_filter in oihw_weights:
                    #shape of single_filter = (64,3,3)
                    vals.append(tf.math.reduce_sum(tf.math.abs(single_filter)))
                filters_to_prune = np.argsort(vals)[:no_of_channels_to_prune]

                for filters_to_prune in no_of_filters_to_prune:
                    oihw_weights[filters_to_prune] = tf.zeros([converted_shape[1], converted_shape[2], converted_shape[3]])
                    mask[channel_to_prune] = tf.zeros([converted_shape[1], converted_shape[2], converted_shape[3]])

                 # shape = (128,64,3,3)
                weights[x] = helpers.convert_from_oihw_to_hwio(oihw_weights)
                weights[self.conv_masks[i]] = helpers.convert_from_oihw_to_hwio(mask)
                 # shape = (64,128,3,3)
            self.set_weights(weights)
            return weights
        
        def prune_channels(self, ratio):
            weights = self.get_weights()
            for i, x in enumerate(self.conv_layers):
                # shape = (3,3,64,128)
                vals = []
                iohw_weights = helpers.convert_from_hwio_to_iohw(weights[x])
                iohw_mask = helpers.convert_from_hwio_to_iohw(weights[self.conv_masks[i]])
                # shape = (64,128,3,3)
                converted_shape = iohw_weights.shape
                no_of_channels = converted_shape[0]*converted_shape[1]
                no_of_channels_to_prune = int(np.round(ratio * no_of_channels))
                channels = tf.reshape(iohw_weights, (no_of_channels,converted_shape[2],converted_shape[3])).numpy()
                mask = tf.reshape(iohw_mask, (no_of_channels,converted_shape[2],converted_shape[3])).numpy()
                # shape = (8192,3,3)
                for channel in channels:
                    vals.append(tf.math.reduce_sum(tf.math.abs(channel)))
                channels_to_prune = np.argsort(vals)[:no_of_channels_to_prune]

                for channel_to_prune in channels_to_prune:
                    channels[channel_to_prune] = tf.zeros([converted_shape[2],converted_shape[3]])
                    mask[channel_to_prune] = tf.zeros([converted_shape[2],converted_shape[3]])

                reshaped_mask = tf.reshape(mask, converted_shape)
                reshaped_weights = tf.reshape(channels, converted_shape)
                weights[x] = helpers.convert_from_iohw_to_hwio(reshaped_weights)
                weights[self.conv_masks[i]] = helpers.convert_from_iohw_to_hwio(reshaped_mask)
            self.set_weights(weights)
            return weights
        def prune_dense_layers(self, ratio):
            weights = self.get_weights()
            for i, layer_to_prune in enumerate(self.dense_layers):
                no_of_rows_to_prune = int(np.round(ratio * len(weights[layer_to_prune])))
                vals = []
                for row in weights[layer_to_prune]:
                    vals.append(np.sum(np.abs(row)))
                rows_to_prune = np.argsort(vals)[:no_of_rows_to_prune]
                for row_to_prune in rows_to_prune:
                    weights[layer_to_prune][row_to_prune] = tf.zeros(len(weights[layer_to_prune][row_to_prune]))
                    weights[self.dense_masks[i]][row_to_prune] = tf.zeros(len(weights[layer_to_prune][row_to_prune]))
            self.set_weights(weights)
            return weights
        
        if structure == 'channel':
            prune_channels(self,ratio)
        if structure == 'filter':
            prune_filter(self,ratio)
        
        if prune_dense_layers==True:
            prune_dense_layers(self, ratio)
        self.set_weights(weights)
        return True
    
    
        
    def prune_magnitude_global_struct(self, ratio, prune_dense_layers=False,structure='channel'
                                     ):
        def prune_filters(self, ratio):
            weights = self.get_weights()
            all_filters = []
            all_masks = []
            vals = []
            for i, layer_to_prune in enumerate(self.conv_layers):
                # convert from e.g. (3,3,64,128) to (128,64,3,3)
                oihw_weights = helpers.convert_from_hwio_to_oihw(weights[layer_to_prune])
                oihw_mask = helpers.convert_from_hwio_to_oihw(weights[self.conv_masks[i]])
                converted_shape = oihw_weights.shape
                no_of_filters = converted_shape[0]
                
                #calculate average magnitude for each filter
                vals = vals + [np.sum(np.abs(single_filter)) / np.prod(single_filter.shape) for single_filter in oihw_weights]
                all_filters = list(all_filters) +  list(oihw_weights)
                all_masks = list(all_masks) + list(oihw_mask)
            no_of_filters_to_prune = int(np.round(ratio * len(vals)))
            filters_to_prune = np.argsort(vals)[:no_of_channels_to_prune]
            
            for filter_to_prune in filters_to_prune:
                all_filters[filter_to_prune] = tf.zeros(all_filters[filter_to_prune].shape) 
                all_masks[filter_to_prune] = tf.zeros(all_filters[filter_to_prune].shape) 
            
            z = 0
            for i, layer_to_prune in enumerate(self.conv_layers):
                original_shape = helpers.convert_from_hwio_to_oihw(weights[layer_to_prune]).shape
                pruned_layer = tf.reshape(all_filters[z:z + original_shape[0]], original_shape)
                pruned_mask = tf.reshape(all_masks[z:z + original_shape[0]], original_shape)
                weights[layer_to_prune] = helpers.convert_from_oihw_to_hwio(pruned_layer)
                weights[self.conv_masks[i]] = helpers.convert_from_oihw_to_hwio(pruned_mask)
                z = z + original_shape[0]
            self.set_weights(weights)
            return weights
        
        def prune_channels(self, ratio):
            weights = self.get_weights()
            all_channels = []
            all_masks = []
            vals = []
            for layer_to_prune in self.conv_layers:
                # convert from e.g. (3,3,1,6) to (1,6,3,3)
                iohw_weights = helpers.convert_from_hwio_to_iohw(weights[layer_to_prune])
                converted_shape = iohw_weights.shape
                no_of_channels = helpers.converted_shape[0]*converted_shape[1]
                #convert from (1,6,3,3) to (6,3,3)
                channels = tf.reshape(iohw_weights, (no_of_channels,converted_shape[2],converted_shape[3])).numpy()
                mask = np.ones((no_of_channels,converted_shape[2],converted_shape[3]))
                #calculate average magnitude for each filter
                vals = vals + [np.sum(np.abs(channel)) / np.prod(channel.shape) for channel in channels]
                all_channels = list(all_channels) +  list(channels)
                all_masks = list(all_masks) + list(mask)
            no_of_channels_to_prune = int(np.round(ratio * len(vals)))
            channels_to_prune = np.argsort(vals)[:no_of_channels_to_prune]
            
            for channel_to_prune in channels_to_prune:
                all_channels[channel_to_prune] = tf.zeros(all_channels[channel_to_prune].shape) 
                all_masks[channel_to_prune] = tf.zeros(all_channels[channel_to_prune].shape) 
            
            z = 0
            for i, layer_to_prune in enumerate(self.conv_layers):
                original_shape = helpers.convert_from_hwio_to_iohw(weights[layer_to_prune]).shape
                pruned_layer = tf.reshape(all_channels[z:z + original_shape[0]*original_shape[1]], original_shape)
                pruned_mask = tf.reshape(all_masks[z:z + original_shape[0]*original_shape[1]], original_shape)
                weights[layer_to_prune] = helpers.convert_from_iohw_to_hwio(pruned_layer)
                weights[self.conv_masks[i]] = helpers.convert_from_iohw_to_hwio(pruned_mask)
                z = z + original_shape[0]*original_shape[1]
            self.set_weights(weights)
            return weights
        
        def prune_dense_layers(self, ratio):
            weights = self.get_weights()
            vals = []
            lengths = []
            for layer_to_prune in self.dense_layers:
                lengths.append(weights[layer_to_prune].shape[0])
                vals = vals + [np.sum(np.abs(row)) / len(row) for row in weights[layer_to_prune]]
            no_of_rows_to_prune = int(np.round(ratio * len(vals)))
            rows_to_prune = np.argsort(vals)[:no_of_rows_to_prune]
            for i, layer_to_prune in enumerate(self.dense_layers):
                for row_to_prune in rows_to_prune:
                    if row_to_prune in range(int(np.sum(lengths[:i])), int(np.sum(lengths[:i+1]))):
                        weights[layer_to_prune][row_to_prune - int(np.sum(lengths[:i]))] = tf.zeros(weights[layer_to_prune][row_to_prune - int(np.sum(lengths[:i]))].shape)
                        
                        weights[self.dense_masks[i]][row_to_prune - int(np.sum(lengths[:i]))] = tf.zeros(weights[layer_to_prune][row_to_prune - int(np.sum(lengths[:i]))].shape)                
            self.set_weights(weights)        
            return weights
        if structure == 'filter':
            prune_filters(self, ratio)
        if structure == 'channel':
            prune_channels(self, ratio)
        
        if prune_dense_layers==True:
            prune_dense_layers(self, ratio)

        return True
    
    
    def prune_magnitude_local_unstruct(self, ratio):
        
        def prune_conv_layers_locally(self, ratio):
            weights = self.get_weights()
            for layer_index, layer in enumerate(self.conv_layers):
                #shape = 3,3,64,128
                converted_weights = helpers.convert_from_hwio_to_iohw(weights[layer]).numpy()
                converted_mask = helpers.convert_from_hwio_to_iohw(weights[self.conv_masks[layer_index]]).numpy()
                #shape = 128,64, 3,3
                layer_shape = converted_weights.shape
                flat_weights = converted_weights.flatten()
                flat_masks = converted_mask.flatten()
                no_of_weights_to_prune = int(np.round(ratio * len(flat_weights)))
                indices_to_delete = np.abs(flat_weights).argsort(0)[:no_of_weights_to_prune]
                for idx_to_delete in indices_to_delete:
                    flat_masks[idx_to_delete] = 0
                    flat_weights[idx_to_delete] = 0
                converted_mask = flat_masks.reshape(layer_shape)
                converted_weights = flat_weights.reshape(layer_shape)
                back_converted_mask = helpers.convert_from_iohw_to_hwio(converted_mask)
                back_converted_weights = helpers.convert_from_iohw_to_hwio(converted_weights)
                weights[layer] = back_converted_weights
                weights[self.conv_masks[layer_index]] = back_converted_mask
            self.set_weights(weights)
            return weights
        
        def prune_dense_layers_locally(self, ratio):
            weights = self.get_weights()
            for index, layer in enumerate(self.dense_layers):
                shape = weights[layer].shape
                flat_weights = weights[layer].flatten()
                flat_mask = weights[self.dense_masks[index]].flatten()

                no_of_weights_to_prune = int(np.round(len(flat_weights)*ratio))
                indices_to_delete = np.abs(flat_weights).argsort()[:no_of_weights_to_prune]
                for idx_to_delete in indices_to_delete:
                    flat_mask[idx_to_delete] = 0
                    flat_weights[idx_to_delete] = 0
                mask_reshaped = flat_mask.reshape(shape)
                weights_reshaped = flat_weights.reshape(shape)
                weights[self.dense_masks[index]] = mask_reshaped
                weights[layer] = weights_reshaped
            self.set_weights(weights)
            return weights
        
        prune_conv_layers_locally(self,ratio)
        prune_dense_layers_locally(self,ratio)
        return True
    
    def find_layers_and_masks(self):
        if len(self.conv_layers) != 0:
            return True
        for i, w in enumerate(self.get_weights()):
            print(i ,'/', len(self.get_weights()))
            if len(w.shape) == 4 and w.shape[0] != 1: 
                if np.all([x == 0 or x == 1 for x in w.flatten()[:100]]) == False: 
                    self.conv_layers.append(i)
                else:
                    self.conv_masks.append(i)
            if len(w.shape) == 2: 
                if np.all([x == 0 or x == 1 for x in w.flatten()[:100]]) == False: 
                    self.dense_layers.append(i)
                else:
                    self.dense_masks.append(i)
        return True
        