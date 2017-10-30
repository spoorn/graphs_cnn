import argparse
import sys
import os
import csv
import numpy as np

from example import *
from strong_learner import *
from knn_classifier import *
from ridge_classifier import *
from logistic import *

def load_csv(path):
  data = []
  with open(path) as fl:
    reader = csv.reader(fl, delimiter=' ')
    for row in reader:
      for i in range(len(row)):
        row[i] = float(row[i])
      example = Example(row[1:], int(row[0]), 1)
      data.append(example)
  return data

def pca(train, test, dim):
  print('running pca...')
  cov = (1.0/(train.shape[0]))*np.dot(train.transpose(), train)
  U, s, V = np.linalg.svd(cov)
  train_red = np.dot(train, U[:,:dim])
  test_red = np.dot(test, U[:,:dim])
  return train_red, test_red

def normalize(train, test):
  train_means = np.mean(train, axis=0)
  train -= train_means
  test -= train_means
  return train, test

def generate_vs(k, dim):
  return np.random.normal(0, 1, (k, dim))

def get_rbf_sigma(dataset):
  sigma = 0.0
  for i in xrange(100):
    indices = np.random.choice(dataset.shape[0], 2, replace=False)
    randy_samples = dataset[indices]
    sigma += np.linalg.norm(randy_samples[0]-randy_samples[1])
  sigma /= 100.0
  sigma /= 2.0
  return sigma

def approx_rbf_kernel(x, v, sigma):
  return np.sin(np.dot(v, x)/sigma)

def random_gaussian_projections(train, test, v, sigma):
  train_res = []
  test_res = []
  for i in xrange(train.shape[0]):
    train_res.append(approx_rbf_kernel(train[i], v, sigma))
  for i in xrange(test.shape[0]):
    test_res.append(approx_rbf_kernel(test[i], v, sigma))
  return np.array(train_res), np.array(test_res)

def convert2numpy(train_prev, test_prev):
  Ntrain = len(train_prev)
  Ntest = len(test_prev)
  D = len(train_prev[0].point)
  train = np.empty((Ntrain,D))
  test = np.empty((Ntest,D))
  train_labels = np.empty(Ntrain)
  test_labels = np.empty(Ntest)

  # Note, haven't saved weights here
  for i in range(Ntrain):
    train[i] = np.array(train_prev[i].point)
    train_labels[i] = train_prev[i].label
  for i in range(Ntest):
    test[i] = np.array(test_prev[i].point)
    test_labels[i] = test_prev[i].label
  return train, train_labels, test, test_labels

def convert2examples(train, train_labels, test, test_labels):
  train_res = []
  test_res = []
  for i in range(train.shape[0]):
    train_res.append(Example(train[i], int(train_labels[i]), 1))
  for i in range(test.shape[0]):
    test_res.append(Example(test[i], int(test_labels[i]), 1))
  return train_res, test_res

if __name__ == '__main__':
  print("loading training data...")
  train = load_csv('./../cnn_outputs/train_fc_output_best_2.csv')
  print("done loading training data")
  print("loading test data...")
  test = load_csv('./../cnn_outputs/test_fc_output_best_2.csv')
  print("done loading test data")

  train, train_labels, test, test_labels = convert2numpy(train, test)
  #train, test = normalize(train, test)
  train, test = pca(train, test, 200)
  #sigma = get_rbf_sigma(train)
  #K = 20000
  #v = generate_vs(K, train.shape[1])
  #train, test = random_gaussian_projections(train, test, v, sigma)
  train, test = convert2examples(train, train_labels, test, test_labels)
  K, v, sigma = None, None, None
  #boosted = RidgeSDCAClassifier(train, 2, 0.2, K, v, sigma, approx_rbf_kernel)#StrongLearner(train, 10)
  #boosted = Logistic(train)
  #sub_test = test[:150,:]
  #sub_labels = test_labels[:150]
  #test = test[150:,:]
  #test_labels = test_labels[150:]
  boosted = StrongLearner(train, 10, K, v, sigma, approx_rbf_kernel)
  boosted.train()
  print("-------train error---------")
  boosted.error(train)
  print("-------test error----------")
  boosted.error(test) 

  #gnb = GaussianNaiveBayes(train, 2)
  #gnb.train()
  #print("-------train error---------")
  #gnb.error(train)
  #print("-------test error----------")
  #gnb.error(test)

