import math

from example import *
from learner import *

class GaussianNaiveBayes(Learner):

  # k is number of classes
  def __init__(self, data, k):
    Learner.__init__(self)
    self.data = data
    self.k = k
    self.d = len(self.data[0].point) # dimensionality of data
    self.n = len(self.data) # number of examples
    self.priors = [0 for i in range(self.k)]
    self.means = [[0 for i in range(self.d)] for j in range(self.k)]
    self.variances = [[0 for i in range(self.d)] for j in range(self.k)]

  def train(self):
    print("training naive bayes...")
    total_weight = 0.0 
 
    for example in self.data:
       x = example.point
       y = example.label
       w = example.weight
       if w == None:
         example.weight = w = 1.0
 
       total_weight += w

       self.priors[y] += w
  
      # in each dimension, update mean
       for j in range(self.d): 
         self.means[y][j] += w * x[j]

    # divide "mean" by weights
    for y in range(self.k):
      for j in range(self.d):
        self.means[y][j] /= self.priors[y]

    # estimate variances in each dimension
    for example in self.data:
      x = example.point
      y = example.label
      w = example.weight

      for j in range(self.d):
        self.variances[y][j] += (w * (x[j] - self.means[y][j]) ** 2)
   
    # divide "variance" by weights 
    for y in range(self.k):
      for j in range(self.d):    
        self.variances[y][j] /= self.priors[y]

    # normalize priors
    for y in range(self.k):
      self.priors[y] /= total_weight

    print("done training naive bayes")

  def log_gaussian_density(self, mu, sigma, x):
    if sigma == 0.0:
      return 0.0
    log_norm_constant = -0.5 * math.log(2 * math.pi * (sigma ** 2))
    z = (x - mu) / sigma
    log_kernel = -0.5 * (z ** 2)
    return log_norm_constant + log_kernel

  def log_prob_in_class(self, example, y):
    log_prob = math.log(self.priors[y])
    for j in range(self.d):
      mu = self.means[y][j]
      sigma = math.sqrt(self.variances[y][j])
      x = example.point[j]
      log_prob += self.log_gaussian_density(mu, sigma, x)
    return log_prob

  def classify(self, example):
    log_probs = [self.log_prob_in_class(example, y) for y in range(self.k)]
    best_label = -1
    best_log_prob = float("-inf")
    for i in range(self.k):
      if log_probs[i] > best_log_prob:
        best_log_prob = log_probs[i]
        best_label = i
    return best_label

