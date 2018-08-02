#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Modified Tensorflow "Deep MNIST for Experts" tutorial to do iceberg detection.

Created on Wed Nov  8 09:20:31 2017
@author: asherhensley

Revisions
3.0     7/5/18      Hensley     Generalize to n-layers
2.0     7/3/18      Hensley     Go from 2 to 3 convolution layers
1.0     7/3/18      Hensley     General clean up

  MIT License
  
  Permission is hereby granted, free of charge, to any person obtaining a 
  copy of this software and associated documentation files (the 
  "Software"), to deal in the Software without restriction, including 
  without limitation the rights to use, copy, modify, merge, publish, 
  distribute, sublicense, and/or sell copies of the Software, and to 
  permit persons to whom the Software is furnished to do so, subject to 
  the following conditions:

  The above copyright notice and this permission notice shall be included 
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""

# **************
# Packages
# **************

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf


# **************
# Config Setup
# **************

class stru:
    def __init__(self):
        self.n_epoch = 0
        self.n_skip = 0
        self.n_folds = 0
        self.rand_seed = 0
        self.im_size = 0
        self.im_trim = 0
        self.batch_size = 0
        self.p_dropout = 0
        self.data_path = ''
        self.ksize = []
        self.layer_depth = []
        self.max_pool_flag = []
        self.fc_nodes = []
        self.im_show = False
        self.field = ''

#Setup
Config = stru()
Config.n_epoch = 5000
Config.n_skip = 10    
Config.n_folds = 5
Config.rand_seed = 13
Config.im_size = 75
Config.im_trim = 15
Config.batch_size = 100
Config.p_dropout = 0.5
Config.data_path = 'data/train.json'
Config.ksize = [3,3,3,3]
Config.layer_depth = [64,64,64,64]
Config.max_pool_flag = [True,True,True,True]
Config.fc_nodes = [128]
Config.im_show = True
Config.field = 'avg'
        

# **************
# Functions
# **************

def get_data(train_set,n,ptr,Config):
    M = np.size(train_set,0)
    N = np.square(get_im_trim_size(Config))   
    im_vec = get_data_type(train_set,ptr,Config.field)
    im_trim = get_im_trim(im_vec,Config)     
    Xdata = np.reshape(im_trim,(1,N))
    start = ptr + 1
    stop = ptr + n
    r = np.array([[1,0],[0,1]])
    ydata = np.zeros((n,2))
    idx = 0
    ydata[idx,:] = r[train_set.is_iceberg.values[ptr]]
    for i in range(start,stop):
        ptr += 1
        idx += 1
        ptr = np.mod(ptr,M)
        im_vec = get_data_type(train_set,ptr,Config.field)
        im_trim = get_im_trim(im_vec,Config)  
        Xdata = np.append(Xdata,np.reshape(im_trim,(1,N)),axis=0)
        ydata[idx,:] = r[train_set.is_iceberg.values[ptr]]
    return Xdata,ydata,ptr

def get_data_type(train_set,ptr,field):
    if field=='avg':
        im_vec1 = getattr(train_set.iloc[ptr],'band_1')
        im_vec2 = getattr(train_set.iloc[ptr],'band_2')
        im_vec = ( np.array(im_vec1) + np.array(im_vec2) ) / 2.0
    else:
        im_vec = getattr(train_set.iloc[ptr],field)
    return im_vec

def cross_validation_split(train,Config):
    N = np.size(train.index)
    np.random.seed(Config.rand_seed)
    x,I = np.unique(np.random.uniform(0,1000,N),return_index=True)
    I = I.reshape([N,1])
    folds = np.zeros([N,1])
    fs = N/Config.n_folds
    ptr = 0
    for i in range(Config.n_folds):
        folds[ptr:ptr+fs+1] = i
        ptr += fs + 1
    return I,folds

def weight_variable(shape,SEED):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=SEED)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
def im_trim_mask(Config):
    J = range(Config.im_trim,Config.im_size-Config.im_trim)
    return J

def get_num_output_pixels(im_size,nlayers,max_pool_flag):
    npixels = im_size
    for i in range(nlayers):
        if max_pool_flag[i]==1:
            npixels = np.ceil(npixels/2.0)
    return int(np.square(npixels))

def get_im_trim(im_vec,Config):
    im_mat = np.reshape(im_vec,(Config.im_size,Config.im_size))
    J = im_trim_mask(Config)
    im_mat = im_mat[J,:]
    im_trim = im_mat[:,J]
    return im_trim

def get_im_trim_size(Config):
    return Config.im_size - Config.im_trim * 2
 
    
# **************
# Data Setup
# **************
    
# Set Random Seeds
tf.set_random_seed(Config.rand_seed)
np.random.seed(Config.rand_seed)
    
# Storage
train_accuracy = np.zeros(Config.n_epoch/Config.n_skip)
test_accuracy = np.zeros(Config.n_epoch/Config.n_skip)

# Import Training Data
train = pd.read_json(Config.data_path)
train['inc_angle'] = pd.to_numeric(train['inc_angle'],errors='coerce')

# Cross-Validation Split
I,folds = cross_validation_split(train,Config)
train_set = train[train.index.isin(I[folds>0])]
valid_set = train[train.index.isin(I[folds==0])]

# Debug Plots
if Config.im_show:
    
    #Cross-Val Splits
    fig = plt.figure(1)
    B = Config.n_folds
    plt.hist(folds,B,facecolor='green',edgecolor='black')
    plt.show()
    
    # Example Images
    Config_debug = Config
    Config_debug.im_trim = 0
    fig = plt.figure(1,figsize=(8,8))
    for i in range(16):
        ax = fig.add_subplot(4,4,i+1)
        im_vec = get_data_type(train_set,i,Config_debug.field)
        im_trim = get_im_trim(im_vec,Config_debug)
        ax.imshow(im_trim,cmap='inferno')
    plt.show()


# **************
# CNN Setup
# **************
  
# Inputs and Outputs
im_trim_size = get_im_trim_size(Config)
x = tf.placeholder(tf.float32, shape=[None, np.square(im_trim_size)])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# Reshape Image
x_image = tf.reshape(x, [-1, im_trim_size, im_trim_size, 1])

# Dropout
keep_prob = tf.placeholder(tf.float32)


# **************
# Conv-Layers
# **************

# Configure Input Layer
W_arg = [Config.ksize[0], Config.ksize[0], 1, Config.layer_depth[0]]
W_conv = [weight_variable(W_arg,Config.rand_seed)]
b_conv = [bias_variable([Config.layer_depth[0]])]
h_conv = [tf.nn.relu(conv2d(x_image,W_conv[0]) + b_conv[0])]
if Config.max_pool_flag[0]==1:
    h_pool = [max_pool_2x2(h_conv[0])]
else:
    h_pool = [h_conv[0]]
    
# Configure Additional Layers
n_conv_layers = len(Config.layer_depth)
for i in range(1,n_conv_layers):
    W_arg = [Config.ksize[i], Config.ksize[i], Config.layer_depth[i-1], Config.layer_depth[i]]
    W_conv.append(weight_variable(W_arg,Config.rand_seed))
    b_conv.append(bias_variable([Config.layer_depth[i]]))
    h_conv.append(tf.nn.relu(conv2d(h_pool[i-1], W_conv[i]) + b_conv[i]))
    if Config.max_pool_flag[i]==1:
        h_pool.append(max_pool_2x2(h_conv[i]))
    else:
        h_pool.append(h_conv[i])    


# **************
# Output-Layers
# **************
        
# Get Number of Nodes from Last Conv Layer
n_pixels = get_num_output_pixels(im_trim_size,n_conv_layers,Config.max_pool_flag)
n_nodes = n_pixels * Config.layer_depth[i]

# Configure First Output Layer
h_pool_flat = tf.reshape(h_pool[i], [-1, n_nodes])
W_fc = [weight_variable([n_nodes, Config.fc_nodes[0]],Config.rand_seed)]
b_fc = [bias_variable([Config.fc_nodes[0]])]
h_fc = [tf.nn.relu(tf.matmul(h_pool_flat, W_fc[0]) + b_fc[0])]
h_fc_drop = [tf.nn.dropout(h_fc[0], keep_prob)]

# Configure Additional Output Layers
j = 0
for j in range(1,len(Config.fc_nodes)):
    W_arg = [Config.fc_nodes[j-1], Config.fc_nodes[j]]
    W_fc.append(weight_variable(W_arg,Config.rand_seed))
    b_fc.append(bias_variable([Config.fc_nodes[j]]))
    h_fc.append(tf.nn.relu(tf.matmul(h_fc_drop[j-1], W_fc[j]) + b_fc[j]))
    h_fc_drop.append(tf.nn.dropout(h_fc[j], keep_prob))
    
# Configure Readout Layer
W_out = weight_variable([Config.fc_nodes[j], 2],Config.rand_seed)
b_out = bias_variable([2])
y_conv = tf.matmul(h_fc_drop[j], W_out) + b_out


# **************
# Training
# **************
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
log_loss = tf.losses.log_loss(tf.argmax(y_, 1),tf.argmax(y_conv, 1))

# Validation Set
Xdatav,ydatav,junk = get_data(valid_set,np.size(valid_set,0),0,Config)
Xdata_all,ydata_all,junk = get_data(train_set,np.size(train_set,0),0,Config)

# Run
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  idx = 0
  ptr = 0
  for i in range(Config.n_epoch):
      
    # Get Next Training Batch
    Xdata,ydata,ptr = get_data(train_set,Config.batch_size,ptr,Config)
    ptr += Config.batch_size
    ptr = np.mod(ptr,np.size(train_set,0))
    
    # Print Accuracy
    if i % Config.n_skip == 0:
        
      train_accuracy[idx] = accuracy.eval(feed_dict={
          x: Xdata, y_: ydata, keep_prob: 1.0})
      
      test_accuracy[idx] = accuracy.eval(feed_dict={
          x: Xdatav, y_: ydatav, keep_prob: 1.0})
    
      #loss = log_loss.eval(feed_dict={x: Xdatav, y_: ydatav, keep_prob: 1.0})
      loss = cross_entropy.eval(feed_dict={x: Xdatav, y_: ydatav, keep_prob: 1.0})
      print('step %d, training accuracy %g, test accuracy %g, log_loss %g' 
          % (i, train_accuracy[idx], test_accuracy[idx], loss))
      idx += 1
    
    # Weight Update
    train_step.run(feed_dict={x: Xdata, y_: ydata, keep_prob: 1-Config.p_dropout})


# **************
# Plot Results
# **************
    
fig = plt.figure(1,figsize=(6,6)) 
ax = fig.add_subplot(1,1,1)  
sup = np.array(range(0,len(train_accuracy))) * Config.n_skip
ax.plot(sup,100*(1-train_accuracy))
ax.plot(sup,100*(1-test_accuracy))   
ax.grid()
ax.legend(["Train","Test"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Classifier Error (%)")
plt.show()
    





