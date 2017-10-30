from example import *
from learner import *
from gnb import *
from knn_classifier import *
from logistic import *
from ridge_classifier import *

class StrongLearner(Learner):

   def __init__(self, data, T, K, v, sigma, h_fn):
     self.data = data
     self.T = T
     self.learners = []
     self.weights = []
     self.K = K
     self.v = v
     self.sigma = sigma
     self.h_fn = h_fn

   def update_weights(self, learner, alpha):
     for example in self.data:
       y = self.rademacherize(example.label)
       y_hat = self.rademacherize(learner.classify(example))
       example.weight = example.weight * math.exp(-alpha * y * y_hat)

   def rademacherize(self, x):
     if x == 0:
       return -1
     return 1

   # adaboost
   def train(self):
     print("beginning boosting...")
     for example in self.data:
       example.weight = 1.0 / len(self.data)
     
     for t in range(self.T):
       print("iteration " + str(t + 1) + " of " + str(self.T) + " of boosting")
       #gnb = GaussianNaiveBayes(self.data, 2)
       #gnb = KnnClassifier(self.data, 2, 3)
       gnb = Logistic(self.data)
       #gnb = RidgeSDCAClassifier(self.data, 2, 0.1, self.K, self.v, self.sigma, self.h_fn) 
       print("dimension d=" + str(gnb.d))
       gnb.train(20, 0.1, 0.001)
       #gnb.train(20, 40, 1.0/18)
       epsilon = gnb.error(self.data, uniform=False)
       alpha = 0.5 * math.log((1 - epsilon) / epsilon)
       self.update_weights(gnb, alpha)
       self.learners.append(gnb)
       self.weights.append(alpha)
 
     print("done boosting") 

   def classify(self, example):
     result = 0.0
     for i in range(len(self.learners)):
       learner = self.learners[i]
       weight = self.weights[i]
       prediction = self.rademacherize(learner.classify(example))
       result += weight * prediction
     if result >= 0:
       return 1
     else:
       return 0

