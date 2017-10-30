# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import argparse
import sys
import os
import csv

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import sklearn as sk
from sklearn import metrics
from scipy.ndimage.interpolation import rotate

from PIL import Image
from PIL import ImageChops


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
  W_conv1 = weight_variable([3, 3, 1, 48], 'W_conv1')
  b_conv1 = bias_variable([48], 'b_conv1')
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([3, 3, 48, 96], 'W_conv2')
  b_conv2 = bias_variable([96], 'b_conv2')
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  #h_pool2 = max_pool_2x2(h_conv2)

  #W_conv3 = weight_variable([5, 5, 64, 128])
  #b_conv3 = bias_variable([128])
  #h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

  # Dimensions for pyramids in SPP
  dims = [1, 2, 3, 4, 5]

  # Spatial Pyramid Pooling
  spp = spp_layer(h_conv2, dims)
  #spp = h_pool2
  fc_dim = sum(x**2 for x in dims)*96
  #fc_dim = 7*7*64

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([fc_dim, 512], 'W_fc1')
  b_fc1 = bias_variable([512], 'b_fc1')

  h_pool3_flat = tf.reshape(spp, [-1, fc_dim])

  # Write spp output to file
  spp_output = tf.reshape(h_pool3_flat, [1, fc_dim])

  h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
  fc_output = tf.reshape(h_fc1, [1, 512])

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32, name='keep_prob')
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map output to 2 classes
  W_fc2 = weight_variable([512, 2], 'W_fc2')
  b_fc2 = bias_variable([2], 'b_fc2')

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob, spp_output, fc_output


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

def weight_variable(shape, nam):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=nam)


def bias_variable(shape, nam):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=nam)

def augmentData(train):
  print("augmenting data...")
  newTrain = []
  for i in range(train.shape[0]):
    newTrain.append(train[i])
    copy = np.copy(train[i]).astype(np.int32)
    if copy[0] == 1:
      iters = 1
    else:
      iters = 1
    for j in range(iters):
       temp = augmentImage(copy)
       newTrain.append(temp)
    #if i % 100 == 0:
      #print("augmented " + str(i))
  newTrain = np.array(newTrain)
  np.random.shuffle(newTrain)
  print("done augmenting")
  return newTrain

# takes an image, and returns an augmentation in same form
def augmentImage(imageAsArray):
  #return imageAsArray
  label = imageAsArray[0]
  w = int(imageAsArray[1])
  h = int(imageAsArray[2])
  matrix = imageAsArray[3:]
  #print("orig: " + str(matrix.shape))
  matrix = matrix.reshape((w,h)).astype(np.int32)
  im = Image.fromarray(matrix) 
  #print("original: " + str(im.size)) 

  # rotation
  rand_rotation = np.random.uniform(0, 360)  
  #im = im.rotate(rand_rotation)
  im = Image.fromarray(rotate(matrix, rand_rotation, reshape=True, mode='constant', cval=255.0).astype(np.int32))
  #print("after rotation: " + str(im.size))
  #im = Image.fromarray(matrix.astype(np.int32))
 
  # translation
  rand_trans_hor = np.random.uniform(-10,10)
  rand_trans_ver = np.random.uniform(-10,10)
  im = ImageChops.offset(im, int(rand_trans_hor), int(rand_trans_ver))
  #print("after translation: " + str(im.size))
 
  # rescaling
  rand_scale = np.random.uniform(1.0/1.6,1.6)
  #rand_scale = np.exp(rand_scale)
  #im = im.resize((int(rand_scale*w), int(rand_scale*h)), Image.ANTIALIAS)
  #print("after rescaling: " + str(im.size))  

  # flipping
  rand_hor_flip = np.random.binomial(1,0.5)
  rand_ver_flip = np.random.binomial(1,0.5)
  if (rand_hor_flip == 1):
    im = im.transpose(Image.FLIP_LEFT_RIGHT)
  if (rand_ver_flip == 1):
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
  #print("after flipping: " + str(im.size))
  # shearing
  rand_shear = np.random.uniform(-20,20)
  

  # stretching
  rand_stretch = np.random.uniform(1.0/1.3, 1.3)
  rand_stretch = np.exp(rand_stretch)
 
  
  # convert back to original form
  w,h = im.size
  arr = np.asarray(im).reshape(w*h)
  #print(arr.shape)
  #np.insert(arr, 0, [label, w, h])
  arr = np.insert(arr, 0, label)
  arr = np.insert(arr, 1, w)
  arr = np.insert(arr, 2, h)
  return arr.astype(np.int32)

# Get training data from file
def getData(path):
  print("loading data...")
  data = []
  with open(path) as fl:
    reader = csv.reader(fl, delimiter=',')
    for row in reader:
      data.append(np.array(row))
  data = np.array(data)
  np.random.shuffle(data)
  print("done loading data")
  train = data#data[375:]
  test = None#data[0:375]
  #train = augmentData(train)
  #train_mean = getMean(train)
  #print("normalizing training images...")
  #train = normalize(train, train_mean)
  #print("normalizing test images...")
  #test = normalize(test, train_mean)
  whiten_dataset(train)
  #whiten_dataset(test)
  return {"train": train, "test": test}

def whiten(image):
  prefix = image[:3]
  image = image[3:].astype(np.int)
  mean = np.mean(image)
  std = np.std(image)
  adj_std = max(std, 1.0 / math.sqrt(image.shape[0]))
  image = (image - mean) / adj_std
  return np.concatenate((prefix, image))

def whiten_dataset(data):
  print("whitening data...")
  for i in xrange(data.shape[0]):
    data[i] = whiten(data[i])
  print("done whitening data")

def getMean(arr):
  print("Calculating mean...")
  res = 0.0
  count = 0
  for i in range(arr.shape[0]):
    res += np.sum(arr[i][3:].astype(int))
    count += arr[i].shape[0]-3
  print("done calculating mean")
  return res/count

# Subtracts 128 from all pixels
def normalize(arr, mean):
  for i in range(arr.shape[0]):
    arr[i].setflags(write=1)
    arr[i][3:] = np.subtract(arr[i][3:].astype(int), mean)
  print("done normalizing images")
  return arr

def main(_):
  # Import data
  data = getData(os.getcwd()+"/data_csvs/128min.csv")
  
  print("creating model...")
  # Create the model
  x = tf.placeholder(tf.float32, [None, None], name='x')

  xtemp = []
  for i in range(1):
    xtemp.append([-1,0,0,1])

  xdims = tf.placeholder_with_default(xtemp, [1,4], name='xdims')

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2], name='y_')

  # Build the graph for the deep net
  y_conv, keep_prob, spp_output, fc_output = deepnn(x, xdims)
 
  cross_entropy = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4, epsilon=1e-5).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='correct_prediction')
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
  y_conv = tf.identity(y_conv, name='y_conv')
  
  print("done creating model")
  print("beginning training...")
  with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    batch_size = 1
    passes = 25
    nTrain = data["train"].shape[0]
    
    for i in range(nTrain*passes): # 50 is good
      if (i+1) % nTrain == 0:
        data["train"] = np.random.permutation(data["train"])
      randy_indices = [i%nTrain]
      randy = data["train"][randy_indices]
      randy_feats = []
      randy_labels = []
      randy_wh = []
      randy_fcdim = 0
      true_label = int(randy[0][0])

      randy_feats.append(randy[0][3:])
      arr = [0,0]
      arr_wh = [-1,0,0,1]
      arr[true_label] = 1
      arr_wh[1] = randy[0][1]
      arr_wh[2] = randy[0][2]
      randy_labels.append(arr)
      randy_wh.append(arr_wh)

      randy_feats = np.array(randy_feats)
      randy_labels = np.array(randy_labels)
      randy_wh = np.array(randy_wh, dtype=np.int32)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: randy_feats, y_: randy_labels, xdims: randy_wh, keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: randy_feats, y_: randy_labels, xdims: randy_wh, keep_prob: 0.55})
      #if i >= data["train"].shape[0]*(passes-1):
      #  if i == data["train"].shape[0]*(passes-1):
      #    print('Saving spp layer output to file (training)...')
      #  spp_res = spp_output.eval(feed_dict={x: randy_feats, y_: randy_labels, xdims: randy_wh, keep_prob: 1.0})
      #  fc_res = fc_output.eval(feed_dict={x: randy_feats, y_: randy_labels, xdims: randy_wh, keep_prob: 1.0})
      #  np.savetxt(f_handle, np.insert(spp_res, 0, int(randy[0][0]), axis=1), delimiter=',', fmt=' '.join(['%d'] + ['%f']*spp_res.shape[1]))
      #  np.savetxt(f_handle2, np.insert(fc_res, 0, int(randy[0][0]), axis=1), delimiter=',', fmt=' '.join(['%d'] + ['%f']*fc_res.shape[1]))
    saver = tf.train.Saver()
    saver.save(sess, 'cnn_saved_model/128min_model')
    exit()
    
    # Output training dataset accuracy after training
    acc = 0.0
    ytrain = []
    ytrain_hat = []
    print('\r\nCalculating training accuracy and confusion matrix')
    f_handle = file('./cnn_outputs/train_spp_output_correct0.csv', 'a')
    f_handle2 = file('./cnn_outputs/train_fc_output_2.csv', 'a')
    for k in range(nTrain):
      # Calculate training accuracy after training...
      randy_indices = [k%nTrain]
      randy = data["train"][randy_indices]
      randy_feats = []
      randy_labels = []
      randy_wh = []
      randy_fcdim = 0
      true_label = int(randy[0][0])

      randy_feats.append(randy[0][3:])
      arr = [0,0]
      arr_wh = [-1,0,0,1]
      arr[true_label] = 1
      arr_wh[1] = randy[0][1]
      arr_wh[2] = randy[0][2]
      randy_labels.append(arr)
      randy_wh.append(arr_wh)

      randy_feats = np.array(randy_feats)
      randy_labels = np.array(randy_labels)
      randy_wh = np.array(randy_wh, dtype=np.int32)

      correct = accuracy.eval(feed_dict={x: randy_feats, y_: randy_labels, xdims: randy_wh, keep_prob: 1.0})
      if k == 0:
        print('Saving spp layer output to file (training)...')
      spp_res = spp_output.eval(feed_dict={x: randy_feats, y_: randy_labels, xdims: randy_wh, keep_prob: 1.0})
      fc_res = fc_output.eval(feed_dict={x: randy_feats, y_: randy_labels, xdims: randy_wh, keep_prob: 1.0})
      #np.savetxt(f_handle, np.insert(spp_res, 0, int(randy[0][0]), axis=1), delimiter=',', fmt=' '.join(['%d'] + ['%f']*spp_res.shape[1]))
      np.savetxt(f_handle2, np.insert(fc_res, 0, int(randy[0][0]), axis=1), delimiter=',', fmt=' '.join(['%d'] + ['%f']*fc_res.shape[1]))

      acc += correct
      ytrain.append(true_label)
      if correct == 1:
        ytrain_hat.append(true_label)
      else:
        ytrain_hat.append(1-true_label)
    f_handle.close()
    f_handle2.close()

    print('training accuracy %g' % (acc/nTrain))
    print('training confusion matrix:')
    print(sk.metrics.confusion_matrix(ytrain, ytrain_hat))
   
    # Test set runs
    y = []
    y_hat = []
    acc = 0.0 
    nTest = data["test"].shape[0]
    test_f_handle = file('./cnn_outputs/test_spp_output_correct0.csv', 'a')
    test_f_handle2 = file('./cnn_outputs/test_fc_output_3.csv', 'a')
    print('\r\nRunning test set analysis')
    for k in range(nTest):
      test_feats = []
      test_labels = []
      test_wh = []
      true_label = int(data["test"][k][0])

      test_feats.append(data["test"][k][3:])
      arr = [0,0]
      arr_wh = [-1,0,0,1]
      arr[true_label] = 1
      arr_wh[1] = data["test"][k][1]
      arr_wh[2] = data["test"][k][2]
      test_labels.append(arr)
      test_wh.append(arr_wh)

      test_feats = np.array(test_feats)
      test_labels = np.array(test_labels)
      test_wh = np.array(test_wh, dtype=np.int32)
      correct = accuracy.eval(feed_dict={x: test_feats, y_: test_labels, xdims: test_wh, keep_prob: 1.0})
      if k == 0:
        print('Saving spp layer output to file (testing)...')
      spp_res = spp_output.eval(feed_dict={x: test_feats, y_: test_labels, xdims: test_wh, keep_prob: 1.0})
      fc_res = fc_output.eval(feed_dict={x: test_feats, y_: test_labels, xdims: test_wh, keep_prob: 1.0})
      #np.savetxt(test_f_handle, np.insert(spp_res, 0, int(data["test"][k][0]), axis=1), delimiter=',', fmt=' '.join(['%d'] + ['%f']*spp_res.shape[1]))
      np.savetxt(test_f_handle2, np.insert(fc_res, 0, int(data["test"][k][0]), axis=1), delimiter=',', fmt=' '.join(['%d'] + ['%f']*fc_res.shape[1]))

      acc += correct
      y.append(true_label)
      if (correct == 1):
        y_hat.append(true_label)
      else:
        y_hat.append(1-true_label)
    test_f_handle.close()
    test_f_handle2.close()

    print('test accuracy %g' % (acc/nTest))
    print("Precision", sk.metrics.precision_score(y, y_hat))
    print("Recall", sk.metrics.recall_score(y, y_hat))
    print("f1_score", sk.metrics.f1_score(y, y_hat))
    print("confusion_matrix")
    print(sk.metrics.confusion_matrix(y, y_hat))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
