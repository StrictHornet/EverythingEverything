#-->import libraries used in solution
import numpy as np
import pandas as pd
#import matplotlib as pt #for . pyplot
#-->from sklearn.impute import SimpleImputer
#from sklearn.model_selection import train_test_split
import math


class Perceptron:
    def __init__(self, learning_rate = 0.1, iters = 20):
      self.lr = learning_rate
      self.iters = iters
      self.activation = self.stepwise
      self.activation_test = self.stepwise_test
      self.max_linear_output = 0
      self.min_linear_output = 0

    def fit(self, X, y):
      n_samples, n_features = X.shape

      #initialise weights
      self.weights = np.zeros(n_features)
      self.bias = 0

      j = "class-1"#y[1]; #Initialise j to an instance of a class in dataset to allow for class differentiation and representation
      #print(j, "is represented as the binary digit 1, secondary comparison class is represented as 0")
      y = np.array([1 if i == j else 0 for i in y])# == "class-1"
      #print(y)

      for iter in range(self.iters): #Number of iterations
        for index, Xsample in enumerate(X): #index elements in X and access each sample in X
          linear_output = np.dot(Xsample, self.weights) + self.bias #Calculate dot product
          if linear_output > self.max_linear_output:
            max_linear_output = linear_output
          if linear_output < self.min_linear_output:
            min_linear_output = linear_output
          y_predicted = self.activation(linear_output) #classify sample

          update = self.lr * (y[index] - y_predicted) #(((y_[index] - y_predicted)**2)/2)
          self.weights += update * Xsample
          self.bias += update
    
    def sigmoid(self, X):
      return 1 / (1 + math.exp(-X))

    def predict(self, X):
      #y = []
      #y_e = []
      a = np.array([])
      for index, Xsample in enumerate(X):
        linear_output = np.dot(Xsample, self.weights) + self.bias
        if linear_output >= 0:
          b = self.sigmoid(linear_output)
          print(b)
        if linear_output > self.max_linear_output:
          max_linear_output = linear_output
        if linear_output < self.min_linear_output:
          min_linear_output = linear_output
        prediction = self.activation_test(linear_output)
        #print(prediction)
        a = np.append(a, prediction)
      return a

    def stepwise(self, x):
      return(np.where(x >= 0, 1, 0))
      #return 1 if x >= 0 else 0

    def stepwise_test(self, x):
      #trap min and max number
      #print(x)
      return(np.where(x >= 0, 1, 0))
      #return 1 if x >= 0 else 0

    def accuracy(y_true, y_pred):
      #accuracy = np.sum(y_true == y_pred) / len(y_true)
      accuracy = np.sum(y_true == y_pred) / len(y_true)
      return accuracy


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