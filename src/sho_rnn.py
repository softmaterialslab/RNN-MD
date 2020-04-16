#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount._DEBUG = False
drive.mount('/content/gdrive/')
#!ls /content/gdrive/'My Drive'/Deeplearning/RA_Work/NEW_SHO_EXP/data/data_dump.pk


# In[ ]:


get_ipython().system("ls /content/gdrive/'My Drive'/Deeplearning/RA_Work/NEW_SHO_EXP/")
working_dir = '/content/gdrive/My Drive/Deeplearning/RA_Work/NEW_SHO_EXP/'


# In[ ]:


#Lib imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('default')
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
from sklearn.metrics import confusion_matrix
import sys, os, io, string, shutil, math
import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA 
from sklearn import preprocessing
from IPython.display import display
import scipy.linalg as la
import re
from tabulate import tabulate
from scipy import stats
import pickle
from sklearn.utils import shuffle
import random

tf.__version__


# In[ ]:


with open(working_dir+'/data/sho_2T_dt=0.001.pk', 'rb') as handle:
    (input_list, all_data, training_indexes, testing_indexes) = pickle.load(handle)

print(len(input_list))
print(all_data.shape)
print(len(training_indexes))
print(len(testing_indexes))


# In[ ]:


import matplotlib.pyplot as plt
#%matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')

fig=plt.figure(figsize=(20, 6))
plt.title('Training data')

reduction_factor =10
window_size=5
input_data = []
output = []

for sim_ in training_indexes[0:20]:
  selected_data = all_data[sim_][::reduction_factor,6]  # 6 referes to actual cos solution
  plt.plot(all_data[sim_][::reduction_factor,0], selected_data, label=input_list[sim_], linewidth=1, markersize=3)
  for i in range(window_size, selected_data.shape[0]):
        input_data.append(selected_data[(i-window_size):i])
        output.append(selected_data[i])

plt.xlabel('time')
plt.ylabel('Position')
plt.legend()
#plt.ylim(-2.5, 10)
plt.xlim(0, 40)

input_data = np.array(input_data)
output = np.array(output)
print(input_data.shape)
print(output.shape)


# In[ ]:


#scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
#scaled_data = scaler.fit_transform(all_data_selected.reshape(-1,1))
#sc = preprocessing.MinMaxScaler() # s the probably the most famous scaling algorithm, and follows the following formula for each feature:
#sc = preprocessing.StandardScaler() # assumes your data is normally distributed within each feature
#sc = preprocessing.RobustScaler() # interquartile range, so if there are outliers in the data, you might want to consider the Robust Scaler
#sc = preprocessing.Normalizer() # The normalizer scales each value by dividing each value by its magnitude in n-dimensional space for n number of features.
#arr_transformed = sc.fit_transform(arr_selected)
#scaled_data = scaled_data.reshape(-1,1000,1)
#scaled_data =all_data_selected


# In[ ]:


input_data_suff, output_suff  = shuffle(input_data, output)

train_test_split = 0.80
train_test_split_ = int(input_data_suff.shape[0]*train_test_split)

x_train = input_data_suff[0:train_test_split_].reshape(-1,window_size,1)
x_test = input_data_suff[train_test_split_:].reshape(-1,window_size,1)
y_train = output_suff[0:train_test_split_]
y_test = output_suff[train_test_split_:]

print("input: ", input_data_suff.shape)
print("Output", output_suff.shape)
print("Train input: ", x_train.shape)
print("Train Output", y_train.shape)
print("Test input: ", x_test.shape)
print("Test Output", y_test.shape)


# In[ ]:


# hyper parameters
learningRate = 0.0001
batchSize = 256
dropout_rate=0.1
epochs=100

input_shape = (window_size, 1)   #batchsize, timesteps, input_dim: this is a bad example here timesteps, input_dim are height and width

# Network Parameters
lstmUnits1 =256       # 1st layer number of neurons
lstmUnits2 = 256       # 1st layer number of neurons
output_shape = 1     # 435*7


# In[ ]:


#This is He initializer
initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal', seed=None)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(lstmUnits1, activation=tf.nn.relu, kernel_initializer=initializer, input_shape=input_shape, return_sequences=True, recurrent_dropout=dropout_rate))
model.add(tf.keras.layers.Dropout(rate=dropout_rate))
model.add(tf.keras.layers.LSTM(lstmUnits2, activation=tf.nn.relu, kernel_initializer=initializer, recurrent_dropout=dropout_rate))
model.add(tf.keras.layers.Dropout(rate=dropout_rate))
#model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_initializer=initializer))
model.add(tf.keras.layers.Dense(output_shape, activation=None, kernel_initializer=initializer))

model.compile(loss=tf.keras.metrics.mean_squared_error, 
             optimizer=tf.keras.optimizers.Adam(lr=learningRate))

#history = model.fit(x_train, y_train, epochs=epochs, batch_size = batchSize,verbose = 1, validation_data = (x_test, y_test))
history = model.fit(x_train, y_train, epochs=epochs, batch_size = batchSize, verbose = 1, validation_data = (x_test, y_test))


# In[ ]:


# This is 8 time frames
#model.evaluate(x_test, y_test)
# Save the model as a hdf5 file
tf.keras.models.save_model(model=model,filepath=working_dir+'/models/SHO_10x.HDF5')

fig, ax = plt.subplots(1,1)
ax.plot(history.history['loss'], color='b', label="Training loss")
ax.plot(history.history['val_loss'], color='r', label="validation loss",axes =ax)
plt.yscale('log')
legend = ax.legend(loc='best', shadow=True)

#ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
#ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
#legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:


# Take a look at the model summary
model.summary()


# In[ ]:


sim_ =training_indexes[0]
#sim_ =testing_indexes[18]

selected_data = all_data[sim_][::reduction_factor,6]

actual_output = []
predicted_output = []

for i in range(window_size, selected_data.shape[0]):
  predicted_output.append(model.predict(selected_data[(i-window_size):i].reshape(-1, window_size, 1)))
  actual_output.append(selected_data[i])

actual_output = np.array(actual_output)
predicted_output = np.array(predicted_output).reshape(-1)

# This is to check continous RNN prediction
Only_RNN_predicted_output = []

temp__ = selected_data[0:window_size]
temp__ = np.append(temp__, predicted_output, axis=0)

for i in range(window_size, selected_data.shape[0]):
  Only_RNN_predicted_output.append(model.predict(temp__[(i-window_size):i].reshape(-1, window_size, 1)))

Only_RNN_predicted_output = np.array(Only_RNN_predicted_output).reshape(-1)


print(actual_output.shape)
print(predicted_output.shape)
print(Only_RNN_predicted_output.shape)
#print(predicted_output)

import matplotlib.pyplot as plt
#%matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')

fig=plt.figure(figsize=(16, 6))
plt.title(input_list[sim_])
plt.plot(actual_output,'r+', label='MD_dynamics', linewidth=1, markersize=3, linestyle='dashed')
plt.plot(predicted_output, label='MD-RNN')
plt.plot(Only_RNN_predicted_output, label='continous RNN')

plt.legend()


# In[ ]:


# Load the keras model
model = tf.keras.models.load_model(filepath=working_dir+'/simple_harmonics.HDF5', compile=True)
#y_pred = model.predict(x_test)
#y_pred_classes = model.predict_classes(x_test)
#cm = confusion_matrix(y_test_classes, y_pred_classes)
#print(cm)


# In[ ]:


#sim_ =training_indexes[0]
sim_ =testing_indexes[18]

selected_data = all_data[sim_][::reduction_factor,6]

actual_output = []
predicted_output = []

for i in range(window_size, selected_data.shape[0]):
  predicted_output.append(model.predict(selected_data[(i-window_size):i].reshape(-1, window_size, 1)))
  actual_output.append(selected_data[i])

actual_output = np.array(actual_output)
predicted_output = np.array(predicted_output).reshape(-1)

# This is to check continous RNN prediction
Only_RNN_predicted_output = []

temp__ = selected_data[0:window_size]
temp__ = np.append(temp__, predicted_output, axis=0)

for i in range(window_size, selected_data.shape[0]):
  Only_RNN_predicted_output.append(model.predict(temp__[(i-window_size):i].reshape(-1, window_size, 1)))

Only_RNN_predicted_output = np.array(Only_RNN_predicted_output).reshape(-1)


print(actual_output.shape)
print(predicted_output.shape)
print(Only_RNN_predicted_output.shape)
#print(predicted_output)

import matplotlib.pyplot as plt
#%matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')

fig=plt.figure(figsize=(16, 6))
plt.title(input_list[sim_])
plt.plot(actual_output,'r+', label='MD_dynamics', linewidth=1, markersize=3, linestyle='dashed')
plt.plot(predicted_output, label='MD-RNN')
plt.plot(Only_RNN_predicted_output, label='continous RNN')

plt.legend()

