import arithmetic_datasets as ad

from tensorflow.keras.layers import Input, Lambda, Dense, concatenate
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, Sequential

import matplotlib.pyplot as plt
import random
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Metric

from tensorflow.keras.callbacks import Callback

from tensorflow.keras.models import load_model

import os

def nondense_model():
    model_name = "nondense_model"
    
#     if not os.path.exists(cb_filepath + "/" + model_name):
#         os.makedirs(cb_filepath + "/" + model_name)  
    
#     checkpoints = [cb_filepath + '/' + model_name + "/" + name
#                    for name in os.listdir(cb_filepath + "/" + model_name)]
#     if checkpoints:
#         latest_cp = max(checkpoints, key=os.path.getctime )
#         print('Restoring from', latest_cp)
#         return load_model(latest_cp)
    
    # Input layer of 3 neurons 
    inp = Input(shape=(1,3))
    
    #128 layer
    d2_out = Dense(128)(inp)

    #grab first, 2nd half of the 128 layer
    d2_out_p1 = Lambda(lambda x: x[:,:,0:64])(d2_out)
    d2_out_p2 = Lambda(lambda x: x[:,:,64:128])(d2_out)

    #64 layer(s)
    d3_out = Dense(64)(d2_out_p1)
    d4_out = Dense(64)(d2_out_p2)

    #grab output nodes from both 64 layers
    d5_out = concatenate([d3_out, d4_out])
    
    o = Dense(1)(d5_out)
    
    model = Model(inp, o)
    
    model._name = model_name
    
    model.compile(
        loss="MeanSquaredError",
        metrics=['accuracy']
    )
    
    return model

def dense_model_5L():
    model_name = "dense_model_5L"
    
#     if not os.path.exists(cb_filepath + "/" + model_name):
#         os.makedirs(cb_filepath + "/" + model_name)  
    
#     checkpoints = [cb_filepath + '/' + model_name + "/" + name
#                    for name in os.listdir(cb_filepath + "/" + model_name)]
#     if checkpoints:
#         latest_cp = max(checkpoints, key=os.path.getctime )
#         print('Restoring from', latest_cp)
#         return load_model(latest_cp)
    
    model_5layer = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1024, input_shape=(1,3)),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1)
    ])
    
#     model.name = "dense_model"

    model_5layer._name = model_name
    model_5layer.compile(
        loss="MeanSquaredError",
        metrics=['accuracy'] #Acc not working, in testing
    )

    return model_5layer

def dense_model2():
    model_name = "dense_model2"
    
#     if not os.path.exists(cb_filepath + "/" + model_name):
#         os.makedirs(cb_filepath + "/" + model_name)  
    
#     checkpoints = [cb_filepath + '/' + model_name + "/" + name
#                    for name in os.listdir(cb_filepath + "/" + model_name)]
#     if checkpoints:
#         latest_cp = max(checkpoints, key=os.path.getctime )
#         print('Restoring from', latest_cp)
#         return load_model(latest_cp)
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2048, input_shape=(1,3)))
    
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(1))
    
#     model.name = "dense_model v2"
    model._name = model_name
    model.compile(
        loss="MeanSquaredError",
        metrics = ["accuracy"]
    )
    
    return model

def rnn_model(dictionary):
    n_hidden = 512
    model = tf.keras.Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(layers.Embedding(len(dictionary), 64, input_length=3))
    # model.add(layers.Dense(64, input_shape=(3,10)))
    # Add a LSTM layer with 128 internal units.
    # model.add(layers.LSTM(128))
    new_shape = (3, 1)
    # model.add(layers.Dense(64, input_shape=new_shape))


    rnn_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(n_hidden),tf.keras.layers.LSTMCell(n_hidden)])
    # model.add(layers.RNN(rnn_cell, input_length=n_input))
    model.add(layers.RNN(rnn_cell, input_shape=new_shape))
    # Add a Dense layer with 10 units.
    model.add(layers.Dense(len(dictionary), activation="softmax"))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
                  # Loss function to minimize
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  # List of metrics to monitor
                  metrics=['sparse_categorical_accuracy'])
    
    return lambda : model

def rnn_model_get_dictionary(*args):
    rdx = None
    for a in args:
        a_flat = a.flatten()
        if rdx is None:
            rdx = np.unique(a_flat)
        rdx = np.unique(np.append(rdx, a_flat))

    reverse_dictionary = np.unique(rdx)

    dictionary = {}
    for x in range(len(reverse_dictionary)):
        dictionary[reverse_dictionary[x]] = x
        
    return dictionary, reverse_dictionary