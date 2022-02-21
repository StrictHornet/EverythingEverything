import numpy as np
import pandas as pd
import matplotlib as pt #for . pyplot
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class Perceptron:

    def __init__(self, learning_rate = 0.1, iters = 1000):
      self.lr = learning_rate
      self.iters = iters
      self.activation = self.stepwise

    def fit(self, X, y):
      n_samples, n_features = X.shape

      #initialise weights
      self.weights = np.zeros(n_features)
      self.bias = 0

      y_ = np.array([4 if i >= 3 else 2 for i in y])# == "class-1"

      for _ in range(self.iters): #Number of iterations
        for index, Xsample in enumerate(X):
          linear_output = np.dot(Xsample, self.weights) + self.bias
          y_predicted = self.activation(linear_output)

          update = self.lr * (y_[index] - y_predicted)
          self.weights += update * Xsample
          self.bias += update


    def predict(self, X):
      linear_output = np.dot(X, self.weights) + self.bias
      y = self.activation(linear_output)
      return y

    def stepwise(self, x):
      return(np.where(x >= 0, 4, 2))
      #return 1 if x >= 0 else 0


#Import Data
#Model Data
#Call Perceptron.fit on Data
#Predict

def accuracy(y_true, y_pred):
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

brain = Perceptron(learning_rate = 0.1, iters = 1000)
brain.fit(X_train, y_train)
predictions = brain.predict(X_test)

a = X_test[:, 0:1]
print(a.transpose())
print("Classification predictions", predictions)
print(accuracy(y_test, predictions))

