# Michael Trinh & Alex Tsun
# K Nearest Neighbors Classifier

from learner import *
from example import *
import numpy as np

class KnnClassifier(Learner):
  
  def __init__(self, data, num_classes, K):
    Learner.__init__(self)
    self.data = data
    self.num_classes = num_classes
    self.K = K
    self.d = len(self.data[0].point) # dimensionality of data
    self.n = len(self.data) # number of examples
    self.data_matrix = np.empty((len(self.data), len(self.data[0].point)))
    self.labels = np.empty(len(self.data))
    for i in range(self.data_matrix.shape[0]):
      self.data_matrix[i] = self.data[i].point
      self.labels[i] = self.data[i].label
    #data_weights = np.array([x.weight for x in self.data])
    #data_weights /= np.sum(data_weights)
    #subspace_indices = np.random.choice(self.n, size=int(self.n*2.0/3), replace=False, p=data_weights)
    #self.labels = self.labels[subspace_indices]
    #self.data_matrix = self.data_matrix[subspace_indices]

    # Assuming self.n is an even number
    self.find_best_subspace()

  def find_best_subspace(self):
    N = self.n
    D = self.d
    pop = self.get_population()
    while pop.shape[0] > 1:
      print "here"
      pop = self.hux_cross_over(pop)
      fitness = self.eval_pop(pop)
      ind = np.argpartition(fitness, -pop.shape[0]/2)[-pop.shape[0]/2:]
    self.data_matrix = self.data_matrix[:,ind]
      
  def get_population(self):
    res = []
    for i in xrange(10):
      res.append(np.random.binomial(1, 0.8, size=self.d))
    return np.array(res)

  def hux_cross_over(self, pop):
    res = []
    for i in xrange(pop.shape[0]/2):
      pair = np.random.choice(pop.shape[0], 2, replace=False)
      p1 = pop[pair[0]]
      p2 = pop[pair[1]]
      xor_res = np.array(p1^p2)
      hamming_dist = np.count_nonzero(xor_res)
      if hamming_dist == 0:
        res.append(p1)
        res.append(p2)
        continue
      indices = np.where(xor_res == 1)[0]
      half = np.random.choice(indices, size=hamming_dist/2, replace=False)
      temp = np.copy(p1[half])
      p1[half] = np.copy(p2[half])
      p2[half] = temp
      res.append(p1)
      res.append(p2)
    return np.array(res)

  def eval_pop(self, pop):
    res = []
    for i in xrange(pop.shape[0]):
      indices = np.where(pop[i] == 1)[0]
      err = 0.0
      curr_mat = self.data_matrix[:,indices]
      for j in xrange(self.data_matrix.shape[0]):
        lab = self.clas(curr_mat, Example(curr_mat[j,:], self.labels[j], self.data[j].weight))
        if lab != self.labels[j]:
          err += 1
      res.append(err)
    return res

  def train(self):
    pass

  def clas(self, data_mat, example):
    k_weights = np.zeros(self.num_classes)
    distances = np.linalg.norm(data_mat-example.point, axis=1)
    indices = np.argpartition(distances, self.K)[:self.K]
    knn_labels = self.labels[indices]
    for i in range(knn_labels.shape[0]):
      k_weights[int(knn_labels[i])] += self.data[indices[i]].weight
    return np.argmax(k_weights)

  def classify(self, example):
    k_weights = np.zeros(self.num_classes)
    distances = np.linalg.norm(self.data_matrix-example.point, axis=1)
    indices = np.argpartition(distances, self.K)[:self.K]
    knn_labels = self.labels[indices]
    for i in range(knn_labels.shape[0]):
      k_weights[int(knn_labels[i])] += self.data[indices[i]].weight
    return np.argmax(k_weights)
