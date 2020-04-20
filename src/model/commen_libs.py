#%tensorflow_version 2.x
#%activate tensorflow-gpu
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import time
import tensorflow as tf
import yaml
import sys
import os
tf.keras.backend.set_floatx('float64')
## Fix for Fail to find the dnn implementation
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print(tf.__version__)
            print(tf.test.gpu_device_name())
            print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    except RuntimeError as e:
        print(e)
'''
'''