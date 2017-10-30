import sklearn as sk
from sklearn import metrics
from example import *

class Learner(object):
  
  # Data should be list of examples
  def __init__(self):
    pass

  def train(self):
    pass

  def classify(self, example):
    pass

  def error(self, data, uniform=True):
    print("evaluating error...")
    err = 0.0
    ys = []
    y_hats = []

    if uniform:
      nTest = len(data)
    else: 
      nTest = sum([example.weight for example in data])

    for example in data:
      y = example.label
      w = example.weight
      y_hat = self.classify(example)
      ys.append(y)
      y_hats.append(y_hat)
      if uniform:
        if y != y_hat:
          err += 1.0
      else:
        if y != y_hat:
          err += w
 
    err = err / nTest

    print('err %g' % err)
    print("Precision", sk.metrics.precision_score(ys, y_hats))
    print("Recall", sk.metrics.recall_score(ys, y_hats))
    print("f1_score", sk.metrics.f1_score(ys, y_hats))
    print("confusion_matrix")
    print(sk.metrics.confusion_matrix(ys, y_hats))

    print("done evaluating error")
    return err

