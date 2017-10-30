# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import csv

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None


def deepnn(x, xdims):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, xdims[0])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([3, 3, 1, 48])
  b_conv1 = bias_variable([48])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([3, 3, 48, 96])
  b_conv2 = bias_variable([96])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  #h_pool2 = max_pool_2x2(h_conv2)

  #W_conv3 = weight_variable([5, 5, 64, 128])
  #b_conv3 = bias_variable([128])
  #h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

  #h_pool3 = max_pool_2x2(h_conv3)

  #W_conv4 = weight_variable([5, 5, 128, 256])
  #b_conv4 = bias_variable([256])
  #h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

  # Dimensions for pyramids in SPP
  dims = [1, 2, 4, 8]

  # Spatial Pyramid Pooling
  spp = spp_layer(h_conv2, dims)
  #spp = h_pool2
  fc_dim = sum(x**2 for x in dims)*96
  #fc_dim = 7*7*64

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([fc_dim, 512])
  b_fc1 = bias_variable([512])

  h_pool3_flat = tf.reshape(spp, [-1, fc_dim])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([512, 2])
  b_fc2 = bias_variable([2])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Max Pooling for any patch nxn
def max_pool_nxn(x, dim):
  shape = tf.shape(x)
  h = tf.cast(tf.gather(shape, 1), tf.int32)
  w = tf.cast(tf.gather(shape, 2), tf.int32)
  res = []

  for r in range(dim):
    for c in range(dim):
      h1 = tf.cast(tf.floor(tf.divide(tf.cast(tf.multiply(r,h),tf.float32),tf.cast(dim,tf.float32))), tf.int32)
      h2 = tf.cast(tf.ceil(tf.divide(tf.cast(tf.multiply((r+1),h),tf.float32),tf.cast(dim,tf.float32))), tf.int32)
      w1 = tf.cast(tf.floor(tf.divide(tf.cast(tf.multiply(c,w),tf.float32),tf.cast(dim,tf.float32))), tf.int32)
      w2 = tf.cast(tf.ceil(tf.divide(tf.cast(tf.multiply((c+1),w),tf.float32),tf.cast(dim,tf.float32))), tf.int32)
      region = x[:, h1:h2, w1:w2, :]
      region_max = tf.reduce_max(region, axis=(1,2))
      res.append(region_max)
  return res

# Runs Spatial Pyramid Pooling to get a fixed sized feature vector
# for each image in tensor x
# @param x : input tensor
# @param dims : dimensions for pyramids
# returns feature vector of fixed size
def spp_layer(x, dims):
  pooled = [] 
  for dim in dims:
    pooled += max_pool_nxn(x, dim)
  res = tf.concat(values=pooled, axis=1)
  return res

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Get training data from file
def getData(path):
  data = []
  with open(path) as fl:
    reader = csv.reader(fl, delimiter=',')
    for row in reader:
      data.append(np.array(row))
  data = np.array(data)
  np.random.shuffle(data)
  train = data[500:]
  test = data[0:500]
  train_mean = getMean(train)
  train = normalize(train, train_mean)
  test = normalize(test, train_mean)
  return {"train": train, "test": test}

def getMean(arr):
  res = 0.0
  count = 0
  for i in range(arr.shape[0]):
    res += np.sum(arr[i][3:].astype(int))
    count += arr[i].shape[0]-3
  return res/count

# Subtracts 128 from all pixels
def normalize(arr, mean):
  for i in range(arr.shape[0]):
    arr[i][3:] = np.subtract(arr[i][3:].astype(int), mean)
  return arr

#def normalize(arr, means):
#  labels = arr[:,0:3]
#  standardized_feats = arr[:,3:]-means
#  return np.concatenate((labels, standardized_feats), axis=1)

def main(_):
  # Import data
  data = getData(os.getcwd()+"/training_full_wh.csv")

  # Create the model
  x = tf.placeholder(tf.float32, [None, None])

  xtemp = []
  for i in range(1):
    xtemp.append([-1,0,0,1])

  xdims = tf.placeholder_with_default(xtemp, [1,4], name='xdims')

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x, xdims)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  #train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 1
    for i in range(data["train"].shape[0]*50):
      randy_indices = np.random.choice(data["train"].shape[0], batch_size, replace=False)
      #randy_indices = [i%data["train"].shape[0]]
      randy = data["train"][randy_indices]
      randy_feats = []
      randy_labels = []
      randy_wh = []
      randy_fcdim = 0
      for k in range(batch_size):
        randy_feats.append(randy[k][3:])
        arr = [0,0]
        arr_wh = [-1,0,0,1]
        arr[int(randy[k][0])] = 1
        arr_wh[1] = randy[k][1]
        arr_wh[2] = randy[k][2]
        randy_labels.append(arr)
        randy_wh.append(arr_wh)
      randy_feats = np.array(randy_feats)
      randy_labels = np.array(randy_labels)
      randy_wh = np.array(randy_wh, dtype=np.int32)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: randy_feats, y_: randy_labels, xdims: randy_wh, keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: randy_feats, y_: randy_labels, xdims: randy_wh, keep_prob: 0.5})
    
    acc = 0.0 
    nTest = data["test"].shape[0]
    for k in range(nTest):
      test_feats = []
      test_labels = []
      test_wh = []
      test_feats.append(data["test"][k][3:])
      arr = [0,0]
      arr_wh = [-1,0,0,1]
      arr[int(data["test"][k][0])] = 1
      arr_wh[1] = data["test"][k][1]
      arr_wh[2] = data["test"][k][2]
      test_labels.append(arr)
      test_wh.append(arr_wh)
      test_feats = np.array(test_feats)
      test_labels = np.array(test_labels)
      test_wh = np.array(test_wh, dtype=np.int32)
      acc += accuracy.eval(feed_dict={x: test_feats, y_: test_labels, xdims: test_wh, keep_prob: 1.0})
    #print('test accuracy %g' % accuracy.eval(feed_dict={
    #    x: test_feats, y_: test_labels, xdims: test_wh, keep_prob: 1.0}))
    print('test accuracy %g' % (acc/nTest))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
