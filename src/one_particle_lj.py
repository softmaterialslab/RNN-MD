#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount._DEBUG = False
drive.mount('/content/gdrive/')
#!ls /content/gdrive/'My Drive'/Deeplearning/RA_Work/NEMD_Simulations/all_data/data_dump.pk


# In[ ]:


get_ipython().system("ls /content/gdrive/'My Drive'/Deeplearning/RA_Work/one_particle_LJ")
working_dir = '/content/gdrive/My Drive/Deeplearning/RA_Work/one_particle_LJ'


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


with open(working_dir+'/data/data_dump_single_atom_LJ_100T_1x.pk', 'rb') as handle:
    (input_list, all_data, training_indexes, testing_indexes) = pickle.load(handle)

print(len(input_list))
print(all_data.shape)
print(len(training_indexes))
print(len(testing_indexes))


# In[ ]:


all_data_selected = all_data[:,::10,1:2]
print(all_data_selected.shape)


# In[ ]:


import scipy as sc
sc.stats.describe(all_data_selected.reshape(-1,1))


# In[ ]:


scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
#scaled_data = scaler.fit_transform(all_data_selected.reshape(-1,1))
#sc = preprocessing.MinMaxScaler() # s the probably the most famous scaling algorithm, and follows the following formula for each feature:
#sc = preprocessing.StandardScaler() # assumes your data is normally distributed within each feature
#sc = preprocessing.RobustScaler() # interquartile range, so if there are outliers in the data, you might want to consider the Robust Scaler
#sc = preprocessing.Normalizer() # The normalizer scales each value by dividing each value by its magnitude in n-dimensional space for n number of features.
#arr_transformed = sc.fit_transform(arr_selected)
#scaled_data = scaled_data.reshape(-1,1000,1)
scaled_data =all_data_selected
scaled_data = all_data_critical_selected.reshape(-1,1000,1)
print(scaled_data.shape)


# In[ ]:


window_size=5
input_data = []
output = []
#for sim_ in training_indexes[0:20]:
for sim_ in  range(0, 1):
#for sim_ in range(scaled_data.shape[0]):
    for i in range(window_size, scaled_data.shape[1]):
        input_data.append(scaled_data[sim_, (i-window_size):i, 0])
        output.append(scaled_data[sim_, i, 0])

input_data = np.array(input_data)
output = np.array(output)
print(input_data.shape)
print(output.shape)


# In[ ]:


input_data_suff, output_suff  = shuffle(input_data, output)

train_test_split = 0.95
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
learningRate = 0.001
batchSize = 32
dropout_rate=0.1
epochs=1

input_shape = (window_size, 1)   #batchsize, timesteps, input_dim: this is a bad example here timesteps, input_dim are height and width

# Network Parameters
lstmUnits1 =128       # 1st layer number of neurons
lstmUnits2 = 128       # 1st layer number of neurons
output_shape = 1     # 435*7


# In[ ]:


#This is He initializer
initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal', seed=None)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(lstmUnits1, activation=tf.nn.tanh, kernel_initializer=initializer, input_shape=input_shape, return_sequences=True, recurrent_dropout=dropout_rate))
model.add(tf.keras.layers.Dropout(rate=dropout_rate))
model.add(tf.keras.layers.LSTM(lstmUnits2, activation=tf.nn.tanh, kernel_initializer=initializer, recurrent_dropout=dropout_rate))
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
tf.keras.models.save_model(model=model,filepath=working_dir+'/one_particle_lj_10X.HDF5')

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


# Load the keras model
model = tf.keras.models.load_model(filepath=working_dir+'/one_particle_lj_10X.HDF5', compile=True)
#y_pred = model.predict(x_test)
#y_pred_classes = model.predict_classes(x_test)
#cm = confusion_matrix(y_test_classes, y_pred_classes)
#print(cm)


# In[ ]:


sim_ =training_indexes[28]
#sim_ =testing_indexes[5]
#sim_ = 3
how_many_steps=100
actual_output = []
predicted_output = []

for i in range(window_size, how_many_steps):
  predicted_output.append(model.predict(scaled_data[sim_, (i-window_size):i, 0].reshape(-1, window_size, 1)))
  actual_output.append(scaled_data[sim_, i, 0])

actual_output = np.array(actual_output)
predicted_output = np.array(predicted_output).reshape(-1)

# This is to check continous RNN prediction
Only_RNN_predicted_output = []

temp__ = scaled_data[sim_, 0:window_size, 0]
temp__ = np.append(temp__, predicted_output, axis=0)
temp__.shape

for i in range(window_size, how_many_steps):
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
plt.plot(scaled_data[sim_,0:how_many_steps],'r+', label='MD_dynamics', linewidth=1, markersize=3, linestyle='dashed')
#plt.plot(scaler.inverse_transform(predicted_output.reshape(-1,1)), label='RNN predicted_dynamics')
#plt.plot(scaler.inverse_transform(Only_RNN_predicted_output.reshape(-1,1)), label='continous RNN')
#plt.plot(predicted_output, label='RNN predicted_dynamics')
plt.plot(temp__, label='continous RNN')
plt.legend()

#print(temp__[0:5])
#print(scaled_data[sim_,0:5])


# In[ ]:


#sim_ =training_indexes[3]
sim_ =testing_indexes[5]

actual_output = []
predicted_output = []

for i in range(window_size, 1000):
  predicted_output.append(model.predict(scaled_data[sim_, (i-window_size):i, 0].reshape(-1, window_size, 1)))
  actual_output.append(scaled_data[sim_, i, 0])

actual_output = np.array(actual_output)
predicted_output = np.array(predicted_output).reshape(-1)

# This is to check continous RNN prediction
Only_RNN_predicted_output = []

temp__ = scaled_data[sim_, 0:window_size, 0]
temp__ = np.append(temp__, predicted_output, axis=0)
temp__.shape

for i in range(window_size, 1000):
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
plt.plot(all_data_selected[sim_,window_size:,0],'r+', label='MD_dynamics', linewidth=1, markersize=3, linestyle='dashed')
plt.plot(scaler.inverse_transform(predicted_output.reshape(-1,1)), label='RNN predicted_dynamics')
plt.plot(scaler.inverse_transform(Only_RNN_predicted_output.reshape(-1,1)), label='continous RNN')

plt.legend()


# In[ ]:


sim_ =0

actual_output = []
predicted_output = []
time_data = []
how_many_steps=1000

for i in range(window_size, how_many_steps):
  predicted_output.append(model.predict(all_data_critical_selected[(i-window_size):i, 0].reshape(-1, window_size, 1)))
  actual_output.append(all_data_critical_selected[i, 0])
  time_data.append(all_data_critical_selected_time[i, 0])

actual_output = np.array(actual_output)
predicted_output = np.array(predicted_output).reshape(-1)
time_data = np.array(time_data)

# This is to check continous RNN prediction
Only_RNN_predicted_output = []

temp__ = all_data_critical_selected[0:window_size, 0]
temp__ = np.append(temp__, predicted_output, axis=0)
temp__.shape

for i in range(window_size, how_many_steps):
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
#plt.title(input_list_critical[sim_])
plt.plot( time_data, actual_output,'r+', label='MD_dynamics', linewidth=1, markersize=3, linestyle='dashed')
#plt.plot(scaler.inverse_transform(predicted_output.reshape(-1,1)), label='RNN predicted_dynamics')
#plt.plot(scaler.inverse_transform(Only_RNN_predicted_output.reshape(-1,1)), label='continous RNN')
#plt.plot(predicted_output, label='RNN predicted_dynamics')
plt.plot(time_data, Only_RNN_predicted_output, label='continous RNN')
plt.legend()

np.savetxt(working_dir+'/Lyapunov-data/RNN-shift_vo=0.010000.out', np.column_stack((time_data, actual_output, Only_RNN_predicted_output)), delimiter='\t')
    
fig=plt.figure(figsize=(16, 6))
#plt.title("Error plot: " + input_list_critical[sim_])
plt.plot((actual_output-Only_RNN_predicted_output)**2, label='Sqaured_Pos_error')
plt.legend()


# In[ ]:


#Lyapunov-data

#simulated_result_file = np.loadtxt(working_dir+'/Lyapunov-data/correct.out')
simulated_result_file = np.loadtxt(working_dir+'/Lyapunov-data/shift_vo=0.010000.out')


all_data_critical_selected = simulated_result_file[::10,1:2]
all_data_critical_selected_time = simulated_result_file[::10,0:1]
print(all_data_critical_selected.shape)

