from __future__ import print_function

import os
import subprocess
import shutil
import sys
from PIL import Image
import numpy as np
import csv

LABELED = False

# Convert images to data matrix where each row is an image (pixel values)
def images2matrix(labeled_examples, path, output_file):
  num_imgs = len(labeled_examples)
  count = 0
  tot_arr = []
  for filename,label in labeled_examples.iteritems():
    im = Image.open(os.getcwd()+"/"+path+"/"+filename+".jpg")
    width, height = im.size
    arr = np.array(rgb2gray(np.array(im)).flatten().astype(int), dtype=object)
    if LABELED:
      arr = np.insert(arr, 0, label)
      arr = np.insert(arr, 1, height)
      arr = np.insert(arr, 2, width)
    else:
      arr = np.insert(arr, 0, filename)
      arr = np.insert(arr, 1, height)
      arr = np.insert(arr, 2, width)
    tot_arr.append(arr)
    count += 1
  tot_arr = np.array(tot_arr)
  if os.path.exists(output_file):
    os.remove(output_file)
  f_handle = file(output_file, 'a')
  for i in range(tot_arr.shape[0]):
    temp = tot_arr[i].reshape(1,tot_arr[i].shape[0])
    if LABELED:
      np.savetxt(f_handle, temp, delimiter=',', fmt='%d')
    else:
      np.savetxt(f_handle, temp, delimiter='|', fmt='|'.join(['%s'] + ['%d']*(temp.shape[1]-1)))
  f_handle.close()

# Convert matrix of rgb values from an image, to greyscale
def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Read CSV file containing filename and label 1/0 graph/not graph
# return dictionary representing the data
def getLabeledExamples(path):
  res = {}
  input_file = csv.DictReader(open(path))
  for row in input_file:
    res[row["filename"]] = int(row["label"])
  return res

# Read a TXT file containing all filenames of data we want to classify
def getAllExamples(path):
  res = {}
  with open(path) as f:
    for line in f:
      res[line.strip()] = -1
  return res
    
if __name__ == '__main__':
  if LABELED:
    labeled_examples = getLabeledExamples(os.getcwd()+"/../../data/preprocess_data/train_labels.csv")
  else:
    labeled_examples = getAllExamples(os.getcwd()+"/data_filenames.csv")
  indir = '128min_images'
  outdir = 'data_csvs/128_all.csv'
  if not os.path.exists('data_csvs'):
    os.mkdir('data_csvs')
  print(sys.argv[0] + ": Converting images in", indir, "to matrices (output to " + os.path.abspath(outdir) +")")
  images2matrix(labeled_examples, indir, outdir)
