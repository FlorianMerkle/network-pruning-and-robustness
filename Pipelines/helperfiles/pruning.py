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

from . import helpers

def _prune_random_local_unstruct(model, ratio, weights):
    def prune_conv_layers(model, ratio, weights):
        for i, layer in enumerate(model.conv_layers):
            #shape = 3,3,64,128
            converted_weights = helpers.convert_from_hwio_to_iohw(weights[layer]).numpy()
            converted_mask = helpers.convert_from_hwio_to_iohw(weights[model.conv_masks[i]]).numpy()
            #shape = 128,64, 3,3
            layer_shape = weights[layer].shape
            flat_masks = converted_mask.flatten()
            flat_weights = weights[layer].flatten()
            no_of_weighs_to_prune = int(np.round(ratio * len(flat_weights)))
            non_zero_weights = np.nonzero(flat_weights)[0]
            no_of_weights_to_prune_left = int(no_of_weighs_to_prune - (len(flat_weights) - len(non_zero_weights)) )
            random.shuffle(non_zero_weights)
            indices_to_delete = non_zero_weights[:no_of_weights_to_prune_left]
            for idx_to_delete in indices_to_delete:
                flat_masks[idx_to_delete] = 0
                flat_weights[idx_to_delete] = 0
            back_converted_mask = helpers.convert_from_iohw_to_hwio(converted_mask)
            back_converted_weights = helpers.convert_from_iohw_to_hwio(converted_weights)
            converted_mask = flat_masks.reshape(layer_shape)
            converted_weights = flat_weights.reshape(layer_shape)
            weights[layer] = back_converted_weights
            weights[model.conv_masks[i]] = back_converted_mask
        return weights
        
    
    def prune_dense_layers(model, ratio, weights):
#            for index, weight in enumerate(weights):
        for i, layer in enumerate(model.dense_layers):
#                if index in dense_layer_to_prune:
                shape = weights[layer].shape
                flat_weights = weights[layer].flatten()
                flat_mask = weights[model.dense_masks[i]].flatten()
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
                weights[model.dense_masks[i]] = mask_reshaped
                weights[layer] = weights_reshaped
        return weights
    weights = prune_conv_layers(model, ratio, weights)
    weights = prune_dense_layers(model, ratio, weights)
    return weights

def _prune_magnitude_global_unstruct(model, ratio, weights):

    flat_weights = []
    flat_mask = []
    all_masks = model.conv_masks + model.dense_masks
    for i, x in enumerate(model.conv_layers + model.dense_layers):
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
    for i, x in enumerate(model.conv_layers + model.dense_layers):
        weights[x] = flat_weights[z:z + np.prod(weights[x].shape)].reshape(weights[x].shape)
        weights[all_masks[i]] = flat_mask[z:z + np.prod(weights[x].shape)].reshape(weights[x].shape)
        z = z + np.prod(weights[x].shape)            
    return weights

# rework kernel/filter wise
def _prune_random_local_struct(model, ratio, weights, prune_dense_layers=False, structure='kernel'):
    def prune_filters(model, ratio, weights):
        for i, layer in enumerate(model.conv_layers):
            # shape = (3,3,64,128)
            oihw_weights = helpers.convert_from_hwio_to_oihw(weights[layer])
            oihw_mask = helpers.convert_from_hwio_to_oihw(weights[model.conv_masks[i]])
            #iohw_weights = helpers.convert_from_hwio_to_iohw(weights[layer])
            #iohw_mask = helpers.convert_from_hwio_to_iohw(weights[model.conv_masks[i]])
            converted_shape = oihw_weights.shape
            no_of_filters = converted_shape[0]
            no_of_filters_to_prune = int(np.round(ratio * no_of_filters))
            #print(kernels)
            non_zero_filters = np.nonzero([np.sum(filt) for filt in oihw_weights])[0]
            #print(non_zero_kernels)
            no_of_filters_to_prune_left = no_of_filters_to_prune - (len(oihw_weights) - len(non_zero_filters))
            random.shuffle(non_zero_filters)
            filters_to_prune = non_zero_filters[:no_of_filters_to_prune_left]
            
            for filter_to_prune in filters_to_prune:
                oihw_weights[filter_to_prune] = tf.zeros([converted_shape[1], converted_shape[2],converted_shape[3]])
                oihw_mask[filter_to_prune] = tf.zeros([converted_shape[1], converted_shape[2],converted_shape[3]])

            weights[layer] = helpers.convert_from_oihw_to_hwio(filters)
            weights[model.conv_masks[i]] = helpers.convert_from_oihw_to_hwio(oihw_mask)
        return weights
    def prune_kernels(model, ratio, weights):
        for i, layer in enumerate(model.conv_layers):
            # shape = (3,3,64,128)
            vals = []
            iohw_weights = helpers.convert_from_hwio_to_iohw(weights[layer])
            iohw_mask = helpers.convert_from_hwio_to_iohw(weights[model.conv_masks[i]])
            converted_shape = iohw_weights.shape
            no_of_kernels = converted_shape[0]*converted_shape[1]
            no_of_kernels_to_prune = int(np.round(ratio * no_of_kernels))
            kernels = tf.reshape(iohw_weights, (no_of_kernels,converted_shape[2],converted_shape[3])).numpy()
            #print(kernels)
            non_zero_kernels = np.nonzero([np.sum(kernel) for kernel in kernels])[0]
            #print(non_zero_kernels)
            no_of_kernels_to_prune_left = no_of_kernels_to_prune - (len(kernels) - len(non_zero_kernels))
            random.shuffle(non_zero_kernels)
            kernels_to_prune = non_zero_kernels[:no_of_kernels_to_prune_left]
            mask = tf.reshape(iohw_mask, 
                                (no_of_kernels,converted_shape[2],converted_shape[3])).numpy()

            for kernel_to_prune in kernels_to_prune:
                kernels[kernel_to_prune] = tf.zeros([converted_shape[2],converted_shape[3]])
                mask[kernel_to_prune] = tf.zeros([converted_shape[2],converted_shape[3]])

            reshaped_mask = tf.reshape(mask, converted_shape)
            reshaped_weights = tf.reshape(kernels, converted_shape)
            weights[layer] = helpers.convert_from_iohw_to_hwio(reshaped_weights)
            weights[model.conv_masks[i]] = helpers.convert_from_iohw_to_hwio(reshaped_mask)
        return weights
    def prune_dense_layers(model, ratio, weights):
        for i, layer_to_prune in enumerate(model.dense_layers):
            rows = weights[layer_to_prune]
            no_of_rows_to_prune = int(np.round(ratio * len(weights[layer_to_prune])))
            non_zero_rows = np.nonzero([np.sum(row) for row in rows])[0]
            no_of_rows_to_prune_left = no_of_rows_to_prune - (len(rows) - len(non_zero_rows))
            random.shuffle(non_zero_rows)
            rows_to_prune = non_zero_rows[:no_of_rows_to_prune_left]
            
            for row_to_prune in rows_to_prune:
                weights[layer_to_prune][row_to_prune] = tf.zeros(len(weights[layer_to_prune][row_to_prune]))
                weights[model.dense_masks[i]][row_to_prune] = tf.zeros(len(weights[layer_to_prune][row_to_prune]))
        return weights
    if structure == 'filter':
        weights = prune_filters(model, ratio, weights)
    if structure == 'kernel':
        weights = prune_kernels(model, ratio, weights)
    
    if prune_dense_layers==True:
        weights = prune_dense_layers(model, ratio, weights)
    
    return weights

def _prune_random_global_struct(model, ratio, prune_dense_layers=False):
    raise Warning('Not yet implemented')
    return False

def _prune_magnitude_local_struct(model, ratio, weights, prune_dense_layers=False, structure='kernel'):
    def prune_filters(model, ratio, weights):
        for i, x in enumerate(model.conv_layers):
            # shape = (3,3,64,128)
            vals = []
            oihw_weights = helpers.convert_from_hwio_to_oihw(weights[x])
            oihw_mask = helpers.convert_from_hwio_to_oihw(weights[model.conv_masks[i]])
            # shape = (128,64,3,3)
            converted_shape = oihw_weights.shape
            no_of_filters = converted_shape[0]
            no_of_filters_to_prune = int(np.round(ratio * no_of_kernels))
            for single_filter in oihw_weights:
                #shape of single_filter = (64,3,3)
                vals.append(tf.math.reduce_sum(tf.math.abs(single_filter)))
            filters_to_prune = np.argsort(vals)[:no_of_kernels_to_prune]

            for filters_to_prune in no_of_filters_to_prune:
                oihw_weights[filters_to_prune] = tf.zeros([converted_shape[1], converted_shape[2], converted_shape[3]])
                mask[kernel_to_prune] = tf.zeros([converted_shape[1], converted_shape[2], converted_shape[3]])

                # shape = (128,64,3,3)
            weights[x] = helpers.convert_from_oihw_to_hwio(oihw_weights)
            weights[model.conv_masks[i]] = helpers.convert_from_oihw_to_hwio(mask)
                # shape = (64,128,3,3)
        return weights
    
    def prune_kernels(model, ratio, weights):
        for i, x in enumerate(model.conv_layers):
            # shape = (3,3,64,128)
            vals = []
            iohw_weights = helpers.convert_from_hwio_to_iohw(weights[x])
            iohw_mask = helpers.convert_from_hwio_to_iohw(weights[model.conv_masks[i]])
            # shape = (64,128,3,3)
            converted_shape = iohw_weights.shape
            no_of_kernels = converted_shape[0]*converted_shape[1]
            no_of_kernels_to_prune = int(np.round(ratio * no_of_kernels))
            kernels = tf.reshape(iohw_weights, (no_of_kernels,converted_shape[2],converted_shape[3])).numpy()
            mask = tf.reshape(iohw_mask, (no_of_kernels,converted_shape[2],converted_shape[3])).numpy()
            # shape = (8192,3,3)
            for kernel in kernels:
                vals.append(tf.math.reduce_sum(tf.math.abs(kernel)))
            kernels_to_prune = np.argsort(vals)[:no_of_kernels_to_prune]

            for kernel_to_prune in kernels_to_prune:
                kernels[kernel_to_prune] = tf.zeros([converted_shape[2],converted_shape[3]])
                mask[kernel_to_prune] = tf.zeros([converted_shape[2],converted_shape[3]])

            reshaped_mask = tf.reshape(mask, converted_shape)
            reshaped_weights = tf.reshape(kernels, converted_shape)
            weights[x] = helpers.convert_from_iohw_to_hwio(reshaped_weights)
            weights[model.conv_masks[i]] = helpers.convert_from_iohw_to_hwio(reshaped_mask)
        return weights
    
    def prune_dense_layers(model, ratio, weights):
        for i, layer_to_prune in enumerate(model.dense_layers):
            no_of_rows_to_prune = int(np.round(ratio * len(weights[layer_to_prune])))
            vals = []
            for row in weights[layer_to_prune]:
                vals.append(np.sum(np.abs(row)))
            rows_to_prune = np.argsort(vals)[:no_of_rows_to_prune]
            for row_to_prune in rows_to_prune:
                weights[layer_to_prune][row_to_prune] = tf.zeros(len(weights[layer_to_prune][row_to_prune]))
                weights[model.dense_masks[i]][row_to_prune] = tf.zeros(len(weights[layer_to_prune][row_to_prune]))
        return weights
    
    if structure == 'kernel':
        prune_kernels(model,ratio, weights)
    if structure == 'filter':
        prune_filter(model,ratio, weights)
    
    if prune_dense_layers==True:
        prune_dense_layers(model, ratio, weights)
    return weights


    
def _prune_magnitude_global_struct(model, ratio, weights, prune_dense_layers=False,structure='kernel'):
    def prune_filters(model, ratio, weights):
        all_filters = []
        all_masks = []
        vals = []
        for i, layer_to_prune in enumerate(model.conv_layers):
            # convert from e.g. (3,3,64,128) to (128,64,3,3)
            oihw_weights = helpers.convert_from_hwio_to_oihw(weights[layer_to_prune])
            oihw_mask = helpers.convert_from_hwio_to_oihw(weights[model.conv_masks[i]])
            converted_shape = oihw_weights.shape
            no_of_filters = converted_shape[0]
            
            #calculate average magnitude for each filter
            vals = vals + [np.sum(np.abs(single_filter)) / np.prod(single_filter.shape) for single_filter in oihw_weights]
            all_filters = list(all_filters) +  list(oihw_weights)
            all_masks = list(all_masks) + list(oihw_mask)
        no_of_filters_to_prune = int(np.round(ratio * len(vals)))
        filters_to_prune = np.argsort(vals)[:no_of_kernels_to_prune]
        
        for filter_to_prune in filters_to_prune:
            all_filters[filter_to_prune] = tf.zeros(all_filters[filter_to_prune].shape) 
            all_masks[filter_to_prune] = tf.zeros(all_filters[filter_to_prune].shape) 
        
        z = 0
        for i, layer_to_prune in enumerate(model.conv_layers):
            original_shape = helpers.convert_from_hwio_to_oihw(weights[layer_to_prune]).shape
            pruned_layer = tf.reshape(all_filters[z:z + original_shape[0]], original_shape)
            pruned_mask = tf.reshape(all_masks[z:z + original_shape[0]], original_shape)
            weights[layer_to_prune] = helpers.convert_from_oihw_to_hwio(pruned_layer)
            weights[model.conv_masks[i]] = helpers.convert_from_oihw_to_hwio(pruned_mask)
            z = z + original_shape[0]
        return weights
    
    def prune_kernels(model, ratio):
        all_kernels = []
        all_masks = []
        vals = []
        for layer_to_prune in model.conv_layers:
            # convert from e.g. (3,3,1,6) to (1,6,3,3)
            iohw_weights = helpers.convert_from_hwio_to_iohw(weights[layer_to_prune])
            converted_shape = iohw_weights.shape
            no_of_kernels = helpers.converted_shape[0]*converted_shape[1]
            #convert from (1,6,3,3) to (6,3,3)
            kernels = tf.reshape(iohw_weights, (no_of_kernels,converted_shape[2],converted_shape[3])).numpy()
            mask = np.ones((no_of_kernels,converted_shape[2],converted_shape[3]))
            #calculate average magnitude for each filter
            vals = vals + [np.sum(np.abs(kernel)) / np.prod(kernel.shape) for kernel in kernels]
            all_kernels = list(all_kernels) +  list(kernels)
            all_masks = list(all_masks) + list(mask)
        no_of_kernels_to_prune = int(np.round(ratio * len(vals)))
        kernels_to_prune = np.argsort(vals)[:no_of_kernels_to_prune]
        
        for kernel_to_prune in kernels_to_prune:
            all_kernels[kernel_to_prune] = tf.zeros(all_kernels[kernel_to_prune].shape) 
            all_masks[kernel_to_prune] = tf.zeros(all_kernels[kernel_to_prune].shape) 
        
        z = 0
        for i, layer_to_prune in enumerate(model.conv_layers):
            original_shape = helpers.convert_from_hwio_to_iohw(weights[layer_to_prune]).shape
            pruned_layer = tf.reshape(all_kernels[z:z + original_shape[0]*original_shape[1]], original_shape)
            pruned_mask = tf.reshape(all_masks[z:z + original_shape[0]*original_shape[1]], original_shape)
            weights[layer_to_prune] = helpers.convert_from_iohw_to_hwio(pruned_layer)
            weights[model.conv_masks[i]] = helpers.convert_from_iohw_to_hwio(pruned_mask)
            z = z + original_shape[0]*original_shape[1]
        return weights
    
    def prune_dense_layers(model, ratio):
        vals = []
        lengths = []
        for layer_to_prune in model.dense_layers:
            lengths.append(weights[layer_to_prune].shape[0])
            vals = vals + [np.sum(np.abs(row)) / len(row) for row in weights[layer_to_prune]]
        no_of_rows_to_prune = int(np.round(ratio * len(vals)))
        rows_to_prune = np.argsort(vals)[:no_of_rows_to_prune]
        for i, layer_to_prune in enumerate(model.dense_layers):
            for row_to_prune in rows_to_prune:
                if row_to_prune in range(int(np.sum(lengths[:i])), int(np.sum(lengths[:i+1]))):
                    weights[layer_to_prune][row_to_prune - int(np.sum(lengths[:i]))] = tf.zeros(weights[layer_to_prune][row_to_prune - int(np.sum(lengths[:i]))].shape)
                    
                    weights[model.dense_masks[i]][row_to_prune - int(np.sum(lengths[:i]))] = tf.zeros(weights[layer_to_prune][row_to_prune - int(np.sum(lengths[:i]))].shape) 
        return weights
    if structure == 'filter':
        prune_filters(model, ratio, weights)
    if structure == 'kernel':
        prune_kernels(model, ratio, weights)
    
    if prune_dense_layers==True:
        prune_dense_layers(model, ratio, weights)

    return weights


def _prune_magnitude_local_unstruct(model, ratio, weights):
    
    def prune_conv_layers(model, ratio, weights):
        for layer_index, layer in enumerate(model.conv_layers):
            #shape = 3,3,64,128
            converted_weights = helpers.convert_from_hwio_to_iohw(weights[layer]).numpy()
            converted_mask = helpers.convert_from_hwio_to_iohw(weights[model.conv_masks[layer_index]]).numpy()
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
            weights[model.conv_masks[layer_index]] = back_converted_mask
        return weights
    
    def prune_dense_layers(model, ratio, weights):
        for index, layer in enumerate(model.dense_layers):
            shape = weights[layer].shape
            flat_weights = weights[layer].flatten()
            flat_mask = weights[model.dense_masks[index]].flatten()

            no_of_weights_to_prune = int(np.round(len(flat_weights)*ratio))
            indices_to_delete = np.abs(flat_weights).argsort()[:no_of_weights_to_prune]
            for idx_to_delete in indices_to_delete:
                flat_mask[idx_to_delete] = 0
                flat_weights[idx_to_delete] = 0
            mask_reshaped = flat_mask.reshape(shape)
            weights_reshaped = flat_weights.reshape(shape)
            weights[model.dense_masks[index]] = mask_reshaped
            weights[layer] = weights_reshaped
        return weights
    
    weights = prune_conv_layers(model,ratio, weights)
    weights = prune_dense_layers(model,ratio, weights)
    return weights

def _find_layers_and_masks(model):
    if len(model.conv_layers) != 0:
        return True
    for i, w in enumerate(model.get_weights()):
        print(i ,'/', len(model.get_weights()))
        if len(w.shape) == 4 and w.shape[0] != 1: 
            if np.all([x == 0 or x == 1 for x in w.flatten()[:100]]) == False: 
                model.conv_layers.append(i)
            else:
                model.conv_masks.append(i)
        if len(w.shape) == 2: 
            if np.all([x == 0 or x == 1 for x in w.flatten()[:100]]) == False: 
                model.dense_layers.append(i)
            else:
                model.dense_masks.append(i)
    return True