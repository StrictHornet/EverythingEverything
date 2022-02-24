import numpy as np
import pandas as pd
import matplotlib as pt #for . pyplot
#from sklearn.impute import SimpleImputer
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

      j = y[1]
      print(j, "is represented as the binary digit 1, secondary comparison class is represented as 0")
      y_ = np.array([1 if i == j else 0 for i in y])# == "class-1"
      #print(y_)

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
      return(np.where(x >= 0, 1, 0))
      #return 1 if x >= 0 else 0


#Import Data
#Model Data
#Call Perceptron.fit on Data
#Predict

def accuracy(y_true, y_pred):
  #accuracy = np.sum(y_true == y_pred) / len(y_true)
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy

dataset = pd.read_csv('traindata.csv')
dataset.columns = ["A", "B", "C", "D", "Class"]

#########################################
"""
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

brain = Perceptron(learning_rate = 0.1, iters = 20)
brain.fit(X_train, y_train)
predictions = brain.predict(X_test)

a = X_test[:, 0:1]
print(a.transpose())
print("Classification predictions", predictions)
print(accuracy(y_test, predictions))
"""
##########################################

#CLASSIFICATION
#Drop all class 1 objects
dataset = dataset[dataset["Class"].str.contains("class-1") == False]
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#print(X_train)
#print(y_train)

brain = Perceptron(learning_rate = 0.9, iters = 80)
brain.fit(X, y)

testset = pd.read_csv('testdat.csv')
testset.columns = ["A", "B", "C", "D", "Class"]
testset = testset[testset["Class"].str.contains("class-1") == False]
X_test = testset.iloc[:, 0:-1].values
y_test = testset.iloc[:, -1].values
y_test = np.array([1 if i == "class-2" else 0 for i in y_test])
predictions = brain.predict(X_test)
print(X_test)
print(predictions)
print(accuracy(y_test, predictions))
print(brain.predict([597, 2.8, 5.1, 1.3]))