import os
import subprocess
import shutil
from PIL import Image
import numpy as np
import csv

def images2matrix(labeled_examples):
  num_imgs = len(labeled_examples)
  d = 28*28+1
  count = 0
  tot_arr = np.empty((num_imgs,d))
  for filename,label in labeled_examples.iteritems():
    im = Image.open(os.getcwd()+"/../../data/resized_images/" + filename + ".jpg")
    arr = np.array(rgb2gray(np.array(im)).flatten().astype(int))
    arr = np.insert(arr, 0, label)
    tot_arr[count] = arr
    count += 1
  np.savetxt("training.csv", tot_arr, delimiter=",", fmt="%d")

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
    
if __name__ == '__main__':
  labeled_examples = getLabeledExamples(os.getcwd()+"/../../data/preprocess_data/train_labels.csv")
  images2matrix(labeled_examples)
