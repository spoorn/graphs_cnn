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


def deepnn(x):
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
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 2])
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
  data = np.array(data).astype(int)
  #data = np.genfromtxt(path, delimiter=',', invalid_raise=False)
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
    res += np.sum(arr[i][1:])
    count += arr[i].shape[0]-1
  return res/count

# Subtracts 128 from all pixels
def normalize(arr, mean):
  for i in range(arr.shape[0]):
    arr[i][1:] = np.subtract(arr[i][1:], mean)
  #labels = arr[:,0]
  #standardized_feats = arr[:,1:]-means
  #return np.insert(standardized_feats, 0, labels, axis=1)
  return arr

def main(_):
  # Import data
  data = getData(os.getcwd()+"/training.csv")

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 50
    for i in range(7000):
      randy_indices = np.random.choice(data["train"].shape[0], batch_size, replace=False)
      randy = data["train"][randy_indices]
      #randy_feats = np.zeros((batch_size,), dtype=np.int64)
      #randy_labels = np.zeros((batch_size,2), dtype=object)
      randy_feats = []
      randy_labels = []
      for k in range(batch_size):
        #randy_feats[k] = randy[k][1:]
        #randy_labels[k,int(randy[k][0])] = 1
        randy_feats.append(randy[k][1:])
        arr = [0,0]
        arr[int(randy[k][0])] = 1
        randy_labels.append(arr)
      randy_feats = np.array(randy_feats)
      randy_labels = np.array(randy_labels)
      #randy_feats = randy[:,1:]
      #for j in range(batch_size):
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: randy_feats, y_: randy_labels, keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: randy_feats, y_: randy_labels, keep_prob: 0.5})

    nTest = data["test"].shape[0]
    test_feats = []
    test_labels = []
    #test_feats = np.zeros((nTest,), dtype=object)
    #test_labels = np.zeros((nTest,2), dtype=object)
    for k in range(nTest):
      test_feats.append(data["test"][k][1:])
      arr = [0,0]
      arr[int(data["test"][k][0])] = 1
      test_labels.append(arr)
    test_feats = np.array(test_feats)
    test_labels = np.array(test_labels)
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: test_feats, y_: test_labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
