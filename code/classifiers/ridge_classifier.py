# Michael Trinh & Alex Tsun
# Ridge Regression Classifier

from learner import *
from example import *
import numpy as np

class RidgeSDCAClassifier(Learner):

  def __init__(self, data, num_classes, lambduh, K = None, v = None, sigma = None, h_fn = None):
    Learner.__init__(self)
    self.data = data
    self.n = len(self.data)
    self.d = len(self.data[0].point)
    self.K = K
    self.v = v
    self.sigma = sigma
    self.h_fn = h_fn
    self.num_classes = num_classes
    self.lambduh = lambduh
    self.alpha = np.zeros(self.n)
    #self.w = np.zeros(self.d)
    self.w = np.zeros(self.K)
    self.train_max = None
    self.train_min = None
    self.data_matrix = np.empty((self.n, self.K))
    self.labels = np.empty(len(self.data))
    for i in range(self.data_matrix.shape[0]):
      self.data_matrix[i] = self.h_fn(self.data[i].point, self.v, self.sigma)
      self.labels[i] = self.data[i].label

  def train(self, epochs, batch_size, gamma):
    for t in xrange(epochs):
      print("on epoch %d" % t)
      perm_indices = np.random.permutation(self.n)
      for i in xrange(self.n/batch_size):
        #w_update = np.zeros(self.d)
        w_update = np.zeros(self.K)
        for j in perm_indices[i*batch_size:(i+1)*batch_size]:
          #x = self.h_fn(self.data_matrix[j], self.v, self.sigma)
          x = self.data_matrix[j]
          y = self.labels[j]
          x_norm = np.linalg.norm(x)
          dot = np.dot(x, self.w)
          delta_alpha = (y - dot) - self.alpha[j]
          delta_alpha /= (1 + x_norm*x_norm/self.lambduh)
          self.alpha[j] += gamma*delta_alpha
          w_update += x*delta_alpha
          if self.train_max is None or self.train_max < dot:
            self.train_max = dot
          if self.train_min is None or self.train_min > dot:
            self.train_min = dot
        self.w += (gamma/self.lambduh)*w_update

  def classify(self, example):
    #x = self.h_fn(example.point, self.v, self.sigma)
    x = example.point
    if len(x) != self.K:
      x = self.h_fn(example.point, self.v, self.sigma)
    dot = np.dot(self.w, x)
    #print int(dot >= 0.5)
    if dot >= 0.5:
      return 1
    else:
      return 0
