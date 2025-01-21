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
from ..helperfiles.helpers import load_data, initialize_base_model, get_zeros_ratio, train_model, compile_model, bb0_attack, pgd_attack,cw2_attack, plot_hist

AUTOTUNE = tf.data.experimental.AUTOTUNE



def run(structure, method, scope, iterations, architecture, structure_to_prune='filter', prune_dense_layers=False, overwrite=False):
    
    if architecture == 'ResNet':
        ds_train, ds_val, ds_test, attack_images, attack_labels = load_data("imagenette")
    if architecture == 'ResNet8' or architecture == 'VGG':
        ds_train, ds_val, ds_test, attack_images, attack_labels = load_data("cifar10")
    if architecture == 'MLP' or architecture == 'CNN':
        ds_train, ds_val, ds_test, attack_images, attack_labels = load_data("mnist")

    if structure == 'structured':
        experiment_name = f'{architecture}-{method}-{scope}-{structure}-{structure_to_prune}'
    if structure == 'unstructured':
        experiment_name = f'{architecture}-{method}-{scope}-{structure}'
    cols = ['iteration','structure','method','scope','pruning_ratio','accuracy','loss','pgd_linf','cw_l2','bb_l0', 'total_params', 'params_left']
    if overwrite==False:
        try:
            results = pd.read_pickle(f'./final-results/{experiment_name}.pkl')
        except:    
            results = pd.DataFrame(columns=cols, dtype='object')
    if overwrite==True:
        results = pd.DataFrame(columns=cols, dtype='object')
    pgd_success_rates = []
    cw_success_rates = []
    bb0_success_rates = []
    all_accuracies = []

    
    compression_rates = [tf.math.pow(2, x).numpy() for x in range(7)]
    pruning_ratios = [1-1/x for x in compression_rates]

    
    for j in tqdm(range(iterations)):
        accuracies = []
        pgd_success_rate = []
        cw_success_rate = []
        bb0_success_rate = []
        
        try: 
            del model
            print('deleted model')
        except:
            print('no model to delete')
            pass
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        tf.random.set_seed(j)
        
        model = initialize_base_model(architecture, ds_train, j ,experiment_name=experiment_name, lr=1e-3, )

        for index, pruning_ratio in tqdm(enumerate(pruning_ratios)):

                print(f'current pruning ratio is{pruning_ratio}, current iteration is {j}')

                if  method=='random' and scope=='global' and structure=='unstructured':
                    model.prune_random_global_unstruct(pruning_ratio)
                elif  method=='random' and scope=='global' and structure=='structured':
                    model.prune_random_global_struct(pruning_ratio, structure_to_prune=structure_to_prune, prune_dense_layers=prune_dense_layers)
                elif  method=='random' and scope=='local' and structure=='unstructured':
                    model.prune_random_local_unstruct(pruning_ratio)
                elif  method=='random' and scope=='local' and structure=='structured':
                    model.prune_random_local_struct(pruning_ratio, structure_to_prune=structure_to_prune, prune_dense_layers=prune_dense_layers)
                elif  method=='magnitude' and scope=='global' and structure=='unstructured':
                    model.prune_magnitude_global_unstruct(pruning_ratio)
                elif  method=='magnitude' and scope=='global' and structure=='structured':
                    model.prune_magnitude_global_struct(pruning_ratio, structure_to_prune=structure_to_prune, prune_dense_layers=prune_dense_layers)
                elif  method=='magnitude' and scope=='local' and structure=='unstructured':
                    model.prune_magnitude_local_unstruct(pruning_ratio)
                elif  method=='magnitude' and scope=='local' and structure=='structured':
                    model.prune_magnitude_local_struct(pruning_ratio, structure_to_prune=structure_to_prune, prune_dense_layers=prune_dense_layers)
                else:
                    raise ValueError("pruning method invalid")

                zeros_ratio, non_zeros, param_count = get_zeros_ratio(model)
                compile_model(architecture, model)

                hist = train_model(architecture, ds_train, ds_val, model, to_convergence=True)

                zeros_ratio, non_zeros, param_count = get_zeros_ratio(model)
                if architecture == 'ResNet' or architecture == 'ResNet8' or architecture=='VGG':
                    res = model.evaluate(ds_test,verbose=0)
                if architecture == 'CNN' or architecture=='MLP':
                    res = model.evaluate(ds_test[0], ds_test[1],verbose=0)
                plot_hist(hist)

                
                if res[1] > .40:
                    bb0_success = bb0_attack(architecture, model, attack_images, attack_labels)
                else: 
                    bb0_success = 'not successful'
                vals = {
                    'iteration':j,
                    'experiment_name':experiment_name,
                    'structure':structure,
                    'method':method,
                    'scope':scope,
                    'pruning_ratio':pruning_ratio,
                    'accuracy':res[1],
                    'loss':res[0],
                    'pgd_linf':pgd_attack(architecture, model, attack_images, attack_labels),
                    'cw_l2':cw2_attack(architecture, model, attack_images, attack_labels),
                    'bb_l0':bb0_success,
                    'total_params':param_count,
                    'params_left':non_zeros
                }
                results = results.append(pd.DataFrame([vals], index=[0], dtype='object'))
                results.to_pickle(f'./final-results/{experiment_name}.pkl')
                results.to_csv(f'./final-results/{experiment_name}.csv', index=False)
    
    
    results.to_pickle(f'./final-results/{experiment_name}.pkl')
    results.to_csv(f'./final-results/{experiment_name}.csv', index=False)
   


