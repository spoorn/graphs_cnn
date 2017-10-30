# Michael Trinh & Alex Tsun
# Ridge Regression Classifier

from learner import *
from example import *
import numpy as np
import math

class Logistic(Learner):
  
  def sigmoid(self, x):
    if x < 0:
      return math.exp(x) / (1.0 + math.exp(x))
    return 1. / (1. + math.exp(-x))
  
  def __init__(self, data):
    Learner.__init__(self)
    self.data = data
    self.n = len(self.data)
    self.d = len(self.data[0].point)
    self.w = np.zeros(self.d)
    self.weights = np.array([x.weight for x in self.data])
    self.weights /= np.sum(self.weights)

  def train(self, epochs, lambduh, eta):
    for t in xrange(epochs):
      print("on epoch %d" % t)
      #perm_indices = np.random.permutation(self.n)
      for k in xrange(self.n):
        i = np.random.choice(self.n, p=self.weights)
        #x = self.data[perm_indices[i]]
        x = self.data[i]
        error = x.label - self.classify(x) 
        for j in xrange(self.d):
          self.w[j] += eta * (-lambduh * self.w[j] + x.point[j] * error)
  
  def classify(self, example):
    return 1 if self.sigmoid(np.dot(self.w, example.point)) >= 0.5 else 0
