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



AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_data(dataset,ratio='100%'):

    def augment(image,label):
        #image = tf.image.convert_image_dtype(image, tf.float32)
        #image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)) # random rotation
        image = tf.image.random_flip_left_right(image)
        #image = tf.image.random_flip_up_down(image)
        #image = tf.image.random_hue(image, 0.08)
        #image = tf.image.random_saturation(image, 0.6, 1.6)
        #image = tf.image.random_contrast(image, 0.7, 1.3)
        #image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
        image = tf.image.resize_with_crop_or_pad(image, 224+60, 224+60) # Add 60 pixels of padding
        image = tf.image.random_crop(image, size=[224,224,3]) # Random crop back to 28x28
        return image,label
    
    @tf.function
    def load_image(datapoint):
        input_image, label = normalize(datapoint)
        return input_image, label
       
    if dataset=='mnist':
        
        ds, info = tfds.load(name=dataset, with_info=True, split=[f"train[:{ratio}]",f"test[:{ratio}]"])
        ds_train=ds[0]
        ds_test=ds[1]
        
        def normalize(x):
            y = {'image': tf.image.convert_image_dtype(x['image'], tf.float32), 'label': x['label']}
            y = (tf.reshape(y['image'],(28*28,1)), y['label'])
            return y
        ds_test = list(ds_test.map(load_image))
        ds_train = list(ds_train.map(load_image))

        x_train = tf.convert_to_tensor([sample[0] for sample in ds_train])
        y_train = tf.convert_to_tensor([sample[1] for sample in ds_train])
        x_test = tf.convert_to_tensor([sample[0] for sample in ds_test])
        y_test = tf.convert_to_tensor([sample[1] for sample in ds_test])

        return [x_train, y_train], [x_test, y_test], x_test[:1000], y_test[:1000]
        
    if dataset=='imagenette':
        ds, info = tfds.load(name=dataset, with_info=True, split=[f"train[:{ratio}]",f"validation[:{ratio}]"])
        
        ds_train=ds[0]
        ds_test=ds[1]
        def normalize(x):
            y = {'image': tf.image.convert_image_dtype(x['image'], tf.float32), 'label': x['label']}
            y = (tf.image.resize(y['image'], (224,224)), y['label'])
            return y


        num_train_examples= info.splits['train'].num_examples
        BATCH_SIZE = 128

        ds_train = (
            ds_train
            .map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .take(num_train_examples)
            .cache()
            .shuffle(num_train_examples)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        ) 

        ds_test = ds_test.map(
            normalize, )
        ds_test = ds_test.batch(BATCH_SIZE)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)



        attack_set = list(ds[1].map(load_image))[:256]

        attack_images = tf.convert_to_tensor([sample[0] for sample in attack_set])
        attack_labels = tf.convert_to_tensor([sample[1] for sample in attack_set])

        return ds_train, ds_test, attack_images, attack_labels
    
    return False

def compile_model(architecture, model, lr=1e-3):
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy() ,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy'],
        experimental_run_tf_function=True
    )
    return True
    

def initialize_base_model(architecture, ds, index, experiment_name, lr=1e-3, save_weights=False):
    from .models import VGG11
    from .models import ResNet
    from .models import LeNet300_100
    from .models import CNN
    if architecture == 'ResNet':
        model = ResNet()
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy() ,
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=['accuracy'],
            experimental_run_tf_function=True
        )
        model.fit(
            x=ds,
            epochs=1,
        )
    if architecture == 'VGG':
        model = VGG11()
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy() ,
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=['accuracy'],
            experimental_run_tf_function=True
        )
        model.fit(
            x=ds,
            epochs=1,
        )
    if architecture == 'CNN' :
        model = CNN()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy() ,
                      metrics=['accuracy'],
                      experimental_run_tf_function=False
                     )
        model.fit(
            x=ds[0],
            y=ds[1],
            batch_size=64,
            epochs=1,
        )
    if architecture == 'MLP':
        model = LeNet300_100()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy() ,
                      metrics=['accuracy'],
                      experimental_run_tf_function=False
                     )
        model.fit(
            x=ds[0],
            y=ds[1],
            batch_size=64,
            epochs=1,
        )
    
    return model



def train_model(architecture, ds_train, ds_test, model, to_convergence=True, epochs=5):
    if architecture=='CNN' or architecture=='MLP':
        if to_convergence == True:
            epochs=500
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        hist = model.fit(
            x=ds_train[0],
            y=ds_train[1],
            batch_size=64,
            epochs=epochs,
            callbacks=[callback],
            validation_data=(ds_test[0], ds_test[1]),
        )
        return hist
    
    if architecture=='ResNet' or architecture == 'VGG':

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            patience=12,
            monitor='val_loss',
            factor=.3,
            min_lr=1e-5,
            min_delta=0
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
        checkpoint_filepath = '/tmp/checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='max',
            save_best_only=True)

        if to_convergence == True:
            epochs = 150

        hist = model.fit(
            x=ds_train,
            epochs=epochs,
            validation_data=ds_test,
            callbacks=[reduce_lr, early_stopping, model_checkpoint_callback],
        )
        return hist

def print_time(text=''):
    now = datetime.now()
    dt_string = now.strftime("%H:%M:%S")
    print(text, dt_string)

def pgd_attack(architecture, model_to_attack, attack_images, attack_labels):
    print_time(text='starting pgd')
    BATCHSIZE = 32
    fmodel = fb.models.TensorFlowModel(model_to_attack, bounds=(0,1))
    attack = fb.attacks.LinfProjectedGradientDescentAttack()
    if architecture == 'CNN' or architecture == 'MLP':
        adversarials, _, success = attack(
            fmodel,
            attack_images,
            attack_labels,
            epsilons=[x/255 for x in [2,4,8,16,32]]
        )
        del fmodel
        return [np.count_nonzero(eps_res)/len(attack_labels) for eps_res in success]
    
    if architecture == 'ResNet'  or architecture == 'VGG':
        res = [[],[],[],[],[],[]]
        strengths = [0.125,0.25,0.5,1,2,4]
        for i in range(8):
            print_time(text=f'pgd batch {i}')
            adversarials, _, success = attack(
                fmodel,
                attack_images[i*BATCHSIZE:(i+1)*BATCHSIZE],
                attack_labels[i*BATCHSIZE:(i+1)*BATCHSIZE],
                epsilons=[x/255 for x in strengths]
            )
            for j in range(len(strengths)):
                res[j] = res[j]+list(success[j])
        print_time(text='ending pgd')
        del fmodel
        return [np.count_nonzero(eps_res)/len(attack_labels) for eps_res in res]
    return False


def cw2_attack(architecture, model_to_attack, attack_images, attack_labels, eps=[100]):
    print_time(text=f'starting cw')
    BATCHSIZE = 32
    fmodel = fb.models.TensorFlowModel(model_to_attack, bounds=(0,1))
    attack = fb.attacks.L2CarliniWagnerAttack(
        binary_search_steps = 9,
        steps= 5000,
        stepsize = 1,
        confidence = 0,
        initial_const = 100,
        abort_early = True,
    )
    if architecture == 'CNN' or architecture == 'MLP':
        adversarials, _, success = attack(
            fmodel,
            attack_images,
            attack_labels,
            epsilons=eps
        )
        dists = [tf.norm(attack_images[i]-adversarials[0][i]).numpy() for i in range(len(attack_images))]
        del fmodel
        
        return dists, success.numpy().tolist()
    
    if architecture == 'ResNet'  or architecture == 'VGG':
        success = []
        dists = [] 
        for i in range(8):
            print_time(text=f'cw batch {i}')
            attack_batch = attack_images[i*BATCHSIZE:(i+1)*BATCHSIZE]
            attack_batch_labels = attack_labels[i*BATCHSIZE:(i+1)*BATCHSIZE]
            adversarials, _, batch_success = attack(
                fmodel,
                attack_batch,
                attack_batch_labels,
                epsilons=eps
            )
            success = success + list(batch_success)
            dists = dists + [tf.norm(attack_batch[j]-adversarials[0][j]).numpy() for j in range(len(attack_batch))]
        print_time(text=f'ending cw')
        del fmodel
        return dists, success
    return False

def bb0_attack(architecture,model_to_attack, attack_images, attack_labels):
    print_time(text=f'starting bb0')
    fmodel = fb.models.TensorFlowModel(model_to_attack, bounds=(0,1))
    init_attack = fb.attacks.DatasetAttack()
    if architecture == 'CNN' or architecture == 'MLP':
        BATCHSIZE = 1000
        BATCHES = 1
    if architecture == 'ResNet' or architecture == 'VGG':
        BATCHSIZE = 32
        BATCHES = 8
    batches = [
        (attack_images[:BATCHSIZE], attack_labels[:BATCHSIZE]), 
        (attack_images[BATCHSIZE:2*BATCHSIZE], attack_labels[BATCHSIZE:2*BATCHSIZE]),
        (attack_images[2*BATCHSIZE:3*BATCHSIZE], attack_labels[2*BATCHSIZE:3*BATCHSIZE]), 
        (attack_images[3*BATCHSIZE:4*BATCHSIZE], attack_labels[3*BATCHSIZE:4*BATCHSIZE])
    ]

    # create attack that picks adversarials from given dataset of samples
    #init_attack = fb.attacks.DatasetAttack()
    init_attack = fb.attacks.DatasetAttack()

    init_attack.feed(fmodel, batches[0][0])   # feed 1st batch of inputs
    init_attack.feed(fmodel, batches[1][0])   # feed 2nd batch of inputs
    init_attack.feed(fmodel, batches[2][0])   # feed 1st batch of inputs
    init_attack.feed(fmodel, batches[3][0])   # feed 2nd batch of inputs
    attack = fb.attacks.L0BrendelBethgeAttack(binary_search_steps=30, steps=500,lr_num_decay=30, lr=1e7, init_attack=init_attack)

    success = []
    dists = [] 
    
    for i in range(BATCHES):
        print_time(text=f'bb0 batch {i}')
        attack_batch = attack_images[i*BATCHSIZE:(i+1)*BATCHSIZE]
        attack_batch_labels = attack_labels[i*BATCHSIZE:(i+1)*BATCHSIZE]
        adversarials, _, batch_success = attack(
            fmodel,
            attack_batch,
            criterion=fb.criteria.Misclassification(attack_batch_labels),
            epsilons=[None]
        )
        
        success = success + list(batch_success)
        dists = dists + [np.count_nonzero(attack_batch[j]-adversarials[0][j]) for j in range(len(attack_batch))]
    print_time(text=f'ending bb0')
    del fmodel
    return dists, success



def convert_from_hwio_to_iohw(weights_hwio):
    return tf.transpose(weights_hwio, [2, 3, 0, 1])

def convert_from_iohw_to_hwio(weights_iohw):
    return tf.transpose(weights_iohw, [2, 3, 0, 1])

def convert_from_iohw_to_oihw(weights_iohw):
    return tf.transpose(weights_iohw, [1, 0, 2, 3])

def convert_from_oihw_to_iohw(weights_oihw):
    return tf.transpose(weights_oihw, [1, 0, 2, 3])

def convert_from_hwio_to_oihw(weights_hwio):
    return tf.transpose(weights_hwio, [3, 2, 0, 1])

def convert_from_oihw_to_hwio(weights_oihw):
    return tf.transpose(weights_oihw, [2, 3, 1, 0])



def get_zeros_ratio(model, layers_to_examine=None):
    if layers_to_examine==None:
        layers_to_examine = model.dense_masks+model.conv_masks
    weights = model.get_weights()
    all_weights = np.array([])
    for x in layers_to_examine:

        all_weights = np.append(all_weights, weights[x].flatten())
    return np.count_nonzero(all_weights)/len(all_weights), np.count_nonzero(all_weights), len(all_weights)

def plot_hist(hist):
    print(hist)
    # summarize history for accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    try:
    # summarize history for lr
        plt.plot(hist.history['lr'])
        plt.title('model lr')
        plt.ylabel('lr')
        plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    except:
        pass