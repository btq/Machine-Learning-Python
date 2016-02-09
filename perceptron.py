import numpy as np

class Perceptron(object):
   """Perceptron classifier
   Parameters
   eta : float
         learning rate (range 0.0-1.0)
   n_iter : int
            number of passes over the training set

   Attributes
   w_ : 1d-array
        weights post-fit
   errors_ : list
             number of misclassifications in every epoch

   """
   def __init__(self, eta=0.01, n_iter=10):
      self.eta = eta
      self.n_iter = n_iter

   def fit(self,X,y):
      """Fit training data
      Parameters
      X : array-like, shape = [n_samples, n_features]
          training vectors
      y : array-like, shape = [n_samples]

      Returns
      self : object
      """
      self.w_ = np.zeros(1+X.shape[1])
      self.errors_ = []

      for _ in range(self.n_iter):
         errors = 0
         for xi, target in zip(X,y):
            update = self.eta * (target - self.predict(xi))