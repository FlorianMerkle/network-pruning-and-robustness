from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tqdm import tqdm
import matplotlib.pyplot as plt
import pathlib
import os
import helperfiles.helpers as helpers

AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.keras.backend.clear_session()  # For easy reset of notebook state.

tf.__version__
#tf.executing_eagerly()
import multiprocessing

def run():
    ds_train, ds_test, _ ,_ = helpers.load_data('imagenette')

    sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True)
    adam = tf.keras.optimizers.Adam()
    adagrad = tf.keras.optimizers.Adagrad()
    rmsprop = tf.keras.optimizers.RMSprop()
    adadelta = tf.keras.optimizers.Adadelta()
    adamax = tf.keras.optimizers.Adamax()
    optimizers = [adagrad, rmsprop, adadelta, adamax, sgd, adam]

    for optimizer in optimizers:
        process_train = multiprocessing.Process(target=helpers.experimental_create_model_and_train, args=(optimizer, ds_train, ds_test,'none'))
        process_train.start()
        process_train.join()
        
if __name__ == "__main__":
    # execute only if run as a script
    run()