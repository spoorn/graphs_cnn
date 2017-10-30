# Load a saved CNN model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import csv
import math

import tensorflow as tf
import numpy as np
import sklearn as sk
from sklearn import metrics
from scipy.ndimage.interpolation import rotate

from PIL import Image
from PIL import ImageChops
#from 2layer_spp import whiten_dataset, whiten

FLAGS = None

def whiten(image):
  prefix = image[:2]
  image = image[2:].astype(np.int)
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


# Get training data from file
def getData(path):
  print("loading data...")
  # Since strings may be too large
  csv.field_size_limit(sys.maxsize)
  filenames = []
  data = []
  with open(path) as fl:
    reader = csv.reader(fl, delimiter='|')
    for row in reader:
      row = np.array(row)
      filenames.append(row[0])
      data.append(np.array(row[1:], dtype=np.int32))
  filenames = np.array(filenames)
  data = np.array(data)
  permute_indices = np.random.permutation(data.shape[0])
  filenames = filenames[permute_indices]
  data = data[permute_indices]
  print("done loading data")
  whiten_dataset(data)
  return {"data": data, "filenames": filenames}

def getMean(arr):
  print("Calculating mean...")
  res = 0.0
  count = 0
  for i in range(arr.shape[0]):
    res += np.sum(arr[i][2:].astype(int))
    count += arr[i].shape[0]-2
  print("done calculating mean")
  return res/count

# Subtracts 128 from all pixels
def normalize(arr, mean):
  for i in range(arr.shape[0]):
    arr[i].setflags(write=1)
    arr[i][2:] = np.subtract(arr[i][2:].astype(int), mean)
  print("done normalizing images")
  return arr

def main(_):
  # Import data
  data = getData(os.getcwd()+"/preprocess_code/data_csvs/128_all.csv")

  with tf.Session() as sess:
    # Load the saved CNN
    saver = tf.train.import_meta_graph('./cnn_saved_model/128min_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./cnn_saved_model/'))
    
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0') 
    xdims = graph.get_tensor_by_name('xdims:0')
    y_ = graph.get_tensor_by_name('y_:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    correct_prediction = graph.get_tensor_by_name('correct_prediction:0')
    accuracy = graph.get_tensor_by_name('accuracy:0')
    y_conv = graph.get_tensor_by_name('y_conv:0')

    # Output training dataset accuracy after training
    y_hat = []
    nData = data['data'].shape[0]
    print('\r\nCalculating training accuracy and confusion matrix')
    for k in range(nData):
      # Calculate training accuracy after training...
      randy_indices = [k%nData]
      randy = data["data"][randy_indices]
      randy_feats = []
      randy_labels = []
      randy_wh = []

      randy_feats.append(randy[0][2:])
      arr = [0,1]
      arr_wh = [-1,0,0,1]
      arr_wh[1] = randy[0][0]
      arr_wh[2] = randy[0][1]
      randy_labels.append(arr)
      randy_wh.append(arr_wh)

      randy_feats = np.array(randy_feats)
      randy_labels = np.array(randy_labels)
      randy_wh = np.array(randy_wh, dtype=np.int32)

      #print(k)
      #print(data['filenames'][k])

      prediction = y_conv.eval(feed_dict={x: randy_feats, y_: randy_labels, xdims: randy_wh, keep_prob: 1.0})
      prediction = np.argmax(prediction)
      y_hat.append(prediction)

    y_hat = np.array(y_hat)
    num_graphs = np.count_nonzero(y_hat)
    num_nongraphs = len(y_hat)-num_graphs
    print('graphs vs nongraphs: %d - %d', num_graphs, num_nongraphs)
    graphs = np.array(data["filenames"][y_hat.astype(bool)])
    nongraphs = np.array(data["filenames"][np.invert(y_hat.astype(bool))])

    # Save filenames of graphs/nongraphs
    graphs_outdir = 'results/graphs_output.txt'
    nongraphs_outdir = 'results/nongraphs_output.txt'
    if not os.path.exists('results'):
      os.mkdir('results')
    f_handle = file(graphs_outdir, 'w')
    f_handle2 = file(nongraphs_outdir, 'w')
    print("Outputting graphs to", graphs_outdir)
    print("Outputting nongraphs to", nongraphs_outdir)
    np.savetxt(f_handle, graphs, delimiter='\n', fmt='%s')
    np.savetxt(f_handle2, nongraphs, delimiter='\n', fmt='%s')
    f_handle.close()
    f_handle2.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
