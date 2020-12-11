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

class CustomConvLayer(layers.Layer):

#    def __init__(self, weights, mask, biases, stride=1, padding='SAME'):
    def __init__(self, input_channels, output_channels, bias=False, filter_size=3, stride=1, padding='SAME'):
        
        super(CustomConvLayer, self).__init__()
        self.w = self.add_weight(
            shape=(filter_size, filter_size, input_channels, output_channels),
            initializer='glorot_uniform',
            trainable=True,
            name='w'
        )
        self.m = self.add_weight(
            shape=(filter_size, filter_size, input_channels, output_channels),
            initializer='ones',
            trainable=False,
            name='m'
        )
        #self.b = self.add_weight(
        #    shape=(output_channels),
        #    initializer='zeros',
        #    trainable=True,
        #    name='b'
        #)
        self.s = stride
        self.p = padding

        
    def call(self, inputs):
        #print(inputs.shape)
        x = tf.nn.conv2d(inputs, tf.multiply(self.w, self.m), strides=[1, self.s, self.s, 1], padding=self.p)
        #x = tf.nn.bias_add(x, self.b)
        return tf.nn.relu(x)

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
    

#conv2D with bias and relu activation


    



    
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
    

class CustomResNetModel(tf.keras.Model):
    def __init__(self):
        super(CustomResNetModel, self).__init__()
        #self.conv1 = layers.Conv2D(64, 7, strides=(2, 2), padding='same')
        #self.aug = DataAugmentationLayer()
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
        self.dense1 = CustomDenseLayer((512, 1000), True, activation='relu')
        self.dense2 = CustomDenseLayer((1000, 10), activation='softmax')
        #self.conv_layers = []
        #self.conv_masks = []
        #self.dense_layers = []
        #self.dense_masks = []
        self.conv_layers = [0, 2, 5, 14, 17, 26, 29, 44, 47, 56, 59, 74, 77, 86, 89, 104, 107]
        self.conv_masks = [1, 8, 11, 20, 23, 35, 38, 50, 53, 65, 68, 80, 83, 95, 98, 110, 113]
        self.dense_layers = [116, 119]
        self.dense_masks = [118, 121]
        


        
        
    def call(self,inputs, training=False):
        #x = tf.keras.layers.experimental.preprocessing.RandomRotation(.25)(inputs)
        #x = tf.keras.layers.experimental.preprocessing.RandomContrast(.8)(x)
        #x = tf.keras.layers.experimental.preprocessing.RandomFlip()(x)
        #x = tf.keras.layers.experimental.preprocessing.RandomTranslation(.25, .25, interpolation='bilinear')(x)
        #x = self.aug(inputs, training)
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.res_block1(x)
        #x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        #x = self.res_block5(x)
        #x = self.res_block6(x)
        x = self.res_block7(x)
        x = self.res_block8(x)
        #x = self.res_block9(x)
        #x = self.res_block10(x)
        #x = self.res_block11(x)
        #x = self.res_block12(x)
        x = self.res_block13(x)
        x = self.res_block14(x)
        #x = self.res_block15(x)
        x = self.res_block16(x)
        x = self.pool2(x)
        x = self.dense1(x)
        
        return self.dense2(x)
    
    def prune_random_local_unstruct(self, ratio):
        def prune_conv_layers_locally(self, ratio):
            weights = self.get_weights()
            for i, layer in enumerate(self.conv_layers):
                converted_weights = helpers.convert_from_hwio_to_iohw(weights[layer]).numpy()
                converted_mask = helpers.convert_from_hwio_to_iohw(weights[self.conv_masks[i]]).numpy()
                for input_index, input_layer in enumerate(converted_weights):
                    for kernel_index, kernel in enumerate(input_layer):
                        shape = kernel.shape
                        flat_weights = kernel.flatten()
                        flat_masks = converted_mask[input_index][kernel_index].flatten()
                        
                        no_of_weighs_to_prune = int(np.round(ratio * len(flat_weights)))
                        # find unpruned weights
                        non_zero_weights = np.nonzero(flat_masks)[0]
                        # calculate the amount of weights to be pruned this round
                        no_of_weights_to_prune_left = int(no_of_weighs_to_prune - (len(flat_weights) - len(non_zero_weights)) )
                        # shuffle all non-zero weights
                        random.shuffle(non_zero_weights)
                        # and take the indices of the first x weights where x is the number of weights to be pruned this round
                        indices_to_delete = non_zero_weights[:no_of_weights_to_prune_left]
                        
                        for idx_to_delete in indices_to_delete:
                            flat_masks[idx_to_delete] = 0
                            flat_weights[idx_to_delete] = 0
                        converted_mask[input_index][kernel_index] = flat_masks.reshape(shape)
                        converted_weights[input_index][kernel_index] = flat_weights.reshape(shape)
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
    
    
    def prune_random_local_struct(self, ratio):
        def prune_conv_layers(self, ratio):
            weights = self.get_weights()
            for i, x in enumerate(self.conv_layers):

                vals = []
                iohw_weights = helpers.convert_from_hwio_to_iohw(weights[x])
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
                weights[x] = helpers.convert_from_iohw_to_hwio(reshaped_weights)
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
        prune_dense_layers(self, ratio)
        
        return True

    def prune_random_global_struct(self, ratio):
        raise Warning('Not yet implemented')
        return False
    
    def prune_magnitude_local_struct(self, ratio):
        def prune_conv_layers(self, ratio):
            weights = self.get_weights()
            for i, x in enumerate(self.conv_layers):
                vals = []
                iohw_weights = helpers.convert_from_hwio_to_iohw(weights[x])
                iohw_mask = helpers.convert_from_hwio_to_iohw(weights[self.conv_masks[i]])
                converted_shape = iohw_weights.shape
                no_of_channels = converted_shape[0]*converted_shape[1]
                no_of_channels_to_prune = int(np.round(ratio * no_of_channels))
                channels = tf.reshape(iohw_weights, (no_of_channels,converted_shape[2],converted_shape[3])).numpy()
                
                mask = tf.reshape(iohw_mask, (no_of_channels,converted_shape[2],converted_shape[3])).numpy()
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
        weights = self.get_weights()
        weights = prune_conv_layers(self, ratio)
        weights = prune_dense_layers(self, ratio)
        self.set_weights(weights)
        return True
        
    def prune_magnitude_global_struct(self, ratio):
        def prune_conv_layers(self, ratio):
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
                #vals = vals + [np.sum(np.abs(channel)) for channel in channels]
                all_channels = list(all_channels) +  list(channels)
                all_masks = list(all_masks) + list(mask)
            #vals = [np.sum(np.abs(channel)) for channel in all_channels]
            no_of_channels_to_prune = int(np.round(ratio * len(vals)))
            #print('lenght of vals',len(vals))
            #print('number of all channels',no_of_channels)
            #print('channels',no_of_channels_to_prune)
            channels_to_prune = np.argsort(vals)[:no_of_channels_to_prune]
            
            for channel_to_prune in channels_to_prune:
                #print(all_channels[channel_to_prune].shape)
                all_channels[channel_to_prune] = tf.zeros(all_channels[channel_to_prune].shape) 
                all_masks[channel_to_prune] = tf.zeros(all_channels[channel_to_prune].shape) 
            z = 0
            for i, layer_to_prune in enumerate(self.conv_layers):
                original_shape = helpers.convert_from_hwio_to_iohw(weights[layer_to_prune]).shape
                pruned_layer = tf.reshape(all_channels[z:z + original_shape[0]*original_shape[1]], original_shape)
                pruned_mask = tf.reshape(all_masks[z:z + original_shape[0]*original_shape[1]], original_shape)
                weights[layer_to_prune] = helpers.convert_from_iohw_to_hwio(pruned_layer)
                weights[self.conv_masks[i]] = helpers.convert_from_iohw_to_hwio(pruned_mask)
                z = original_shape[0]*original_shape[1]
            self.set_weights(weights)
            return weights
        
        def prune_dense_layers(self, ratio):
            vals = []
            lengths = []
            for layer_to_prune in self.dense_layers:
                #print('dense',layer_to_prune)
                lengths.append(weights[layer_to_prune].shape[0])
                vals = vals + [np.sum(np.abs(row)) / len(row) for row in weights[layer_to_prune]]
                #vals = vals + [np.sum(np.abs(row)) for row in weights[layer_to_prune]]
            no_of_rows_to_prune = int(np.round(ratio * len(vals)))
            #print('rows', no_of_rows_to_prune)
            rows_to_prune = np.argsort(vals)[:no_of_rows_to_prune]
            for i, layer_to_prune in enumerate(self.dense_layers):
                for row_to_prune in rows_to_prune:
                    if row_to_prune in range(int(np.sum(lengths[:i])), int(np.sum(lengths[:i+1]))):
                        weights[layer_to_prune][row_to_prune - int(np.sum(lengths[:i]))] = tf.zeros(weights[layer_to_prune][row_to_prune - int(np.sum(lengths[:i]))].shape)
                        weights[self.dense_masks[i]][row_to_prune - int(np.sum(lengths[:i]))] = tf.zeros(weights[layer_to_prune][row_to_prune - int(np.sum(lengths[:i]))].shape)                
            self.set_weights(weights)        
            return weights
        weights = self.get_weights()
        weights = prune_conv_layers(self, ratio)
        weights = prune_dense_layers(self, ratio)
        #self.set_weights(weights)
        return True
    
    
    def prune_magnitude_local_unstruct(self, ratio):
        def prune_conv_layers_locally(self, ratio):

            #print('inside conv prune func',get_zeros_ratio(self.get_weights()))
            weights = self.get_weights()
            #for w in weights:
                #print(w.shape)
            for layer_index, layer in enumerate(self.conv_layers):
                #print(layer)
                mask = self.conv_masks[layer_index]

                converted_weights = helpers.convert_from_hwio_to_iohw(weights[layer]).numpy()
                converted_mask = helpers.convert_from_hwio_to_iohw(weights[mask]).numpy()
            
                #print('convert weights',converted_weights.shape)
                for input_index, input_layer in enumerate(converted_weights):
                    #print(input_index, '/', len(converted_weights))
                    for kernel_index, kernel in enumerate(input_layer):
                        shape = kernel.shape
                        #print('kernel',shape)
                        flat_weights = kernel.flatten()
                        flat_masks = converted_mask[input_index][kernel_index].flatten()
                        #flat_weights_df = pd.DataFrame(flat_weights)
                        #flat_mask_df = pd.DataFrame(flat_masks)
                        no_of_weights_to_prune = int(np.round(len(flat_weights)*ratio))
                        #print('weights to prune',no_of_weights_to_prune)
                        #print('total weights here', np.round(len(flat_weights)))
                        #indices_to_delete = flat_weights_df.abs().values.argsort(0)[:no_of_weights_to_prune]
                        indices_to_delete = np.abs(flat_weights).argsort(0)[:no_of_weights_to_prune]
                        #print('flat weights shape',flat_weights.shape)
                        #print('flat_masks shape',flat_masks.shape)
                        #print('indices to delete and amount of weights to prune',indices_to_delete, no_of_weights_to_prune)
                        for idx_to_delete in indices_to_delete:
                            
                            flat_masks[idx_to_delete] = 0
                            flat_weights[idx_to_delete] = 0

                        converted_mask[input_index][kernel_index] = flat_masks.reshape(shape)
                        converted_weights[input_index][kernel_index] = flat_weights.reshape(shape)
                back_converted_mask = helpers.convert_from_iohw_to_hwio(converted_mask)
                back_converted_weights = helpers.convert_from_iohw_to_hwio(converted_weights)
                weights[layer] = back_converted_weights
                weights[mask] = back_converted_mask
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
        
        weights = prune_conv_layers_locally(self,ratio)
        weights = prune_dense_layers_locally(self,ratio)
        return True
    
    def find_layers_and_masks(self):
        if len(self.conv_layers) != 0:
            return True
        for i, w in enumerate(self.get_weights()):
            print(i ,'/', len(self.get_weights()))
            if len(w.shape) == 4 and w.shape[0] != 1: 
                if np.all([x == 0 or x == 1 for x in w.flatten()]) == False: 
                    self.conv_layers.append(i)
                else:
                    self.conv_masks.append(i)
            if len(w.shape) == 2: 
                if np.all([x == 0 or x == 1 for x in w.flatten()]) == False: 
                    self.dense_layers.append(i)
                else:
                    self.dense_masks.append(i)
        return True
    