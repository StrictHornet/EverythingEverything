# -->import libraries used in solution
import numpy as np
import pandas as pd
import math


class Perceptron:
    def __init__(self, learning_rate=0.1, iters=20):
        self.lr = learning_rate
        self.iters = iters
        self.activation = self.stepwise

    def fit(self, X, y):
        n_features = X.shape[1]

        # initialise weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        for iter in range(self.iters):  # Number of iterations
            # index elements in X and access each sample in X
            for index, Xsample in enumerate(X):
                # Calculate dot product
                linear_output = np.dot(Xsample, self.weights) + self.bias
                y_predicted = self.activation(linear_output)  # classify sample
            #print(y[index], y_predicted)
                # (((y_[index] - y_predicted)**2)/2)
                update = self.lr * (y[index] - y_predicted)
                self.weights = self.weights + (update * Xsample)
                self.bias = self.bias + update

    def sigmoid(self, X):
        return 1 / (1 + math.exp(-X))

    def predict(self, X):
        X_class = np.array([])
        confidence_arr = np.array([])
        for index, Xsample in enumerate(X):
            linear_output = np.dot(Xsample, self.weights) + self.bias
            prediction = self.activation(linear_output)
            X_class = np.append(X_class, prediction)

        return X_class

    def multiclass_predict(self, X):
        len = X.shape[0]
        X_class = np.empty(shape=(len, 2))
        #X_class = np.array([[0, 0]])
        #X_class = np.array([])
        for index, Xsample in enumerate(X):
            linear_output = np.dot(Xsample, self.weights) + self.bias
            # print(linear_output)
            prediction = self.activation(linear_output)
            #prediction = 100
            # sigmoid function for multiclass classification
            if linear_output >= 0:
                confidence = self.sigmoid(linear_output)
                #X_class = np.append(X_class, confidence)
            else:
                confidence = self.sigmoid(linear_output)
                #confidence_arr = np.append(confidence_arr, confidence)

            X_class[index] = [prediction, confidence]
            #X_class = np.append(X_class, prediction)
        return X_class

    def stepwise(self, x):
        return(np.where(x >= 0, 1, 0))


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def multiclass_accuracy(y_true, y_pred):
    i = 0
    sum = 0
    for element in y_pred:
        actual = y_true[i]
        prediction = y_pred[i]
        i += 1
        if actual == prediction:
            #print(actual, prediction)
            sum += 1
            accuracy = sum / len(y_true)
    return accuracy


def maxConfidence(training_predictions_1vR, training_predictions_2vR, training_predictions_3vR):
    ypred = np.array([])
    len = training_predictions_1vR.shape[0]
    for i in range(len):
        if (training_predictions_1vR[i][0] > training_predictions_2vR[i][0]) and (training_predictions_1vR[i][0] > training_predictions_3vR[i][0]):
            ypred = np.append(ypred, "class-1")
            continue

        if (training_predictions_2vR[i][0] > training_predictions_1vR[i][0]) and (training_predictions_2vR[i][0] > training_predictions_3vR[i][0]):
            ypred = np.append(ypred, "class-2")
            continue

        if (training_predictions_3vR[i][0] > training_predictions_1vR[i][0]) and (training_predictions_3vR[i][0] > training_predictions_2vR[i][0]):
            ypred = np.append(ypred, "class-3")
            continue

        if training_predictions_1vR[i][0] == training_predictions_2vR[i][0]:
            if training_predictions_1vR[i][1] > training_predictions_2vR[i][1]:
                ypred = np.append(ypred, "class-1")
            else:
                ypred = np.append(ypred, "class-2")
            continue

        if training_predictions_1vR[i][0] == training_predictions_3vR[i][0]:
            if training_predictions_1vR[i][1] > training_predictions_3vR[i][1]:
                ypred = np.append(ypred, "class-1")
            else:
                ypred = np.append(ypred, "class-3")
            continue

        if training_predictions_2vR[i][0] == training_predictions_3vR[i][0]:
            if training_predictions_2vR[i][1] > training_predictions_3vR[i][1]:
                ypred = np.append(ypred, "class-2")
            else:
                ypred = np.append(ypred, "class-3")
            continue

    return ypred

##############################
#QUESTION 3 START
##############################
print("QUESTION 3:\n")
#Set Shuffle Seed
np.random.seed(71)

#Import Training and Test Data
train_data = pd.read_csv('train.data', header=None)
test_data = pd.read_csv('test.data', header=None)

train_data = np.array(train_data.values)
test_data = np.array(test_data.values)

#Splitting Test and Training Data
class1_train_data = train_data[0:40, :]
class2_train_data = train_data[40:80, :]
class3_train_data = train_data[80:, :]

#Creating Train Subset Datasets of all comparative classes
class_1v2_train = np.concatenate((class1_train_data, class2_train_data))
class_1v3_train = np.concatenate((class1_train_data, class3_train_data))
class_2v3_train = np.concatenate((class2_train_data, class3_train_data))

##############################
#CLASS 1 VS CLASS 2 START
##############################
#Shuffling Class 1 and 2 for splitting into X,y
np.random.shuffle(class_1v2_train)

#Splitting Class 1 and 2 into X,y
X = class_1v2_train[:, 0:-1]
y = class_1v2_train[:, -1]
y = np.array([1 if i == "class-1" else 0 for i in y])

#Training Classifier on Class 1 and 2
class_1v2_classifier = Perceptron(learning_rate = 0.1, iters = 20)
class_1v2_classifier.fit(X, y)

#Classifying Class 1 and 2 Training Data
training_predictions = class_1v2_classifier.predict(X)
print("The accuracy of the class 1 vs class 2 classifier on training data is", accuracy(y, training_predictions)*100, "%")

#Classifying Class 1 and 2 Test Data
test_data_1v2 = test_data[test_data[:, -1] != "class-3"]
X_test = test_data_1v2[:, 0:-1]
y_test = test_data_1v2[:, -1]
y_test = np.array([1 if i == "class-1" else 0 for i in y_test])
predictions = class_1v2_classifier.predict(X_test)
print("The accuracy of the class 1 vs class 2 classifier on test data is", accuracy(y_test, predictions)*100, "%")
##############################
#CLASS 1 VS CLASS 2 END
##############################

##############################
#CLASS 1 VS CLASS 3 START
##############################
#Shuffling Class 1 and 3 for splitting into X,y
np.random.shuffle(class_1v3_train)

#Splitting Class 1 and 3 into X,y
X = class_1v3_train[:, 0:-1]
y = class_1v3_train[:, -1]
y = np.array([1 if i == "class-1" else 0 for i in y])

#Training Classifier on Class 1 and 3
class_1v3_classifier = Perceptron(learning_rate = 0.1, iters = 20)
class_1v3_classifier.fit(X, y)

#Classifying Class 1 and 3 Training Data
training_predictions = class_1v3_classifier.predict(X)
print("The accuracy of the class 1 vs class 3 classifier on training data is", accuracy(y, training_predictions)*100, "%")

#Classifying Class 1 and 3 Test Data
test_data_1v3 = test_data[test_data[:, -1] != "class-2"]
X_test = test_data_1v3[:, 0:-1]
y_test = test_data_1v3[:, -1]
y_test = np.array([1 if i == "class-1" else 0 for i in y_test])
predictions = class_1v3_classifier.predict(X_test)
print("The accuracy of the class 1 vs class 3 classifier on test data is", accuracy(y_test, predictions)*100, "%")
##############################
#CLASS 1 VS CLASS 3 END
##############################

##############################
#CLASS 2 VS CLASS 3 START
##############################
#Shuffling Class 2 and 3 for splitting into X,y
np.random.shuffle(class_2v3_train)

#Splitting Class 2 and 3 into X,y
X = class_2v3_train[:, 0:-1]
y = class_2v3_train[:, -1]
y = np.array([1 if i == "class-2" else 0 for i in y])

#Training Classifier on Class 2 and 3
class_2v3_classifier = Perceptron(learning_rate = 0.1, iters = 20)
class_2v3_classifier.fit(X, y)

#Classifying Class 2 and 3 Training Data
training_predictions = class_2v3_classifier.predict(X)
print("The accuracy of the class 2 vs class 3 classifier on training data is", accuracy(y, training_predictions)*100, "%")

#Classifying Class 2 and 3 Test Data
test_data_2v3 = test_data[test_data[:, -1] != "class-1"]
#np.random.shuffle(test_data_2v3)
X_test = test_data_2v3[:, 0:-1]
y_test = test_data_2v3[:, -1]
y_test = np.array([1 if i == "class-2" else 0 for i in y_test])
predictions = class_2v3_classifier.predict(X_test)
print("The accuracy of the class 2 vs class 3 classifier on test data is", accuracy(y_test, predictions)*100, "%")
##############################
#CLASS 2 VS CLASS 3 END
##############################

print("\n")
##############################
#QUESTION 3 END
##############################





##############################
#QUESTION 4 START
##############################
print("QUESTION 4:\n")

#Set Shuffle Seed
np.random.seed(71)

#Import Training and Test Data
train_data = pd.read_csv('train.data', header=None)
test_data = pd.read_csv('test.data', header=None)

train_data = np.array(train_data.values)
test_data = np.array(test_data.values)

##############################
#CLASS 1 VS REST START
##############################
#Shuffling Class 1 and REST for splitting into X,y
class_1vR_train = np.copy(train_data)
np.random.shuffle(class_1vR_train)

#Splitting Class 1 and 2 into X,y
X = class_1vR_train[:, 0:-1]
y = class_1vR_train[:, -1]
y = np.array([1 if i == "class-1" else 0 for i in y])

#print(X)

#Training Classifier on Class 1 VS REST
class_1vR_classifier = Perceptron(learning_rate = 0.1, iters = 20)
class_1vR_classifier.fit(X, y)

#Classifying Class 1 on Training Data
training_predictions_1vR = class_1vR_classifier.multiclass_predict(X)
#print("The accuracy of the CLASS 1 vs rest classifier on training data is", multiclass_accuracy(y, training_predictions)*100, "%")
#print(training_predictions)
for i in range(0):
  print(i, training_predictions_1vR[i][0], round(training_predictions_1vR[i][1]*100))
  i += 1


#Classifying Class 1 vs REST Test Data
test_data_1vR = test_data
X_test = test_data_1vR[:, 0:-1]
y_test = test_data_1vR[:, -1]
test_predictions_1vR = class_1vR_classifier.multiclass_predict(X_test)
#print("The accuracy of the CLASS 1 vs REST classifier on test data is", multiclass_accuracy(y_test, predictions)*100, "%")
#print(y_test)
#print(test_predictions_1vR)

##############################
#CLASS 1 VS REST END
##############################


##############################
#CLASS 2 VS REST START
##############################
np.random.seed(71)
#Shuffling Class 2 and REST for splitting into X,y
class_2vR_train = np.copy(train_data)
np.random.shuffle(class_2vR_train)

#Splitting Dataset into X,y
X = class_2vR_train[:, 0:-1]
y = class_2vR_train[:, -1]
y_classlabels = class_2vR_train[:, -1]
y = np.array([1 if i == "class-2" else 0 for i in y])

#print(X)

#Training Classifier on Class 2 VS REST
class_2vR_classifier = Perceptron(learning_rate = 0.1, iters = 20)
class_2vR_classifier.fit(X, y)

#Classifying Class 2 Training Data
training_predictions_2vR = class_2vR_classifier.multiclass_predict(X)
#print("The accuracy of the CLASS 1 vs rest classifier on training data is", multiclass_accuracy(y, training_predictions)*100, "%")
#print(training_predictions)
for i in range(0):
  print(y, i, training_predictions_2vR[i][0], round(training_predictions_2vR[i][1]*100))
  i += 1

#Classifying Class 2 vs REST Test Data
'''
test_data_2vR = test_data
X_test = test_data_2vR[:, 0:-1]
y_test = test_data_2vR[:, -1]
'''
test_predictions_2vR = class_2vR_classifier.multiclass_predict(X_test)
#print("The accuracy of the CLASS 1 vs REST classifier on test data is", multiclass_accuracy(y_test, predictions)*100, "%")
#print(y_test)
#print(test_predictions_1vR)
##############################
#CLASS 2 VS REST END
##############################


##############################
#CLASS 3 VS REST START
##############################
np.random.seed(71)
#Shuffling Class 3 and REST for splitting into X,y
class_3vR_train = np.copy(train_data)
np.random.shuffle(class_3vR_train)

#Splitting Dataset into X,y
X = class_3vR_train[:, 0:-1]
y = class_3vR_train[:, -1]
y_classlabels = class_3vR_train[:, -1]
y = np.array([1 if i == "class-3" else 0 for i in y])

#print(X)

#Training Classifier on Class 3 VS REST
class_3vR_classifier = Perceptron(learning_rate = 0.1, iters = 20)
class_3vR_classifier.fit(X, y)

#Classifying Class 3 Training Data
training_predictions_3vR = class_3vR_classifier.multiclass_predict(X)
#print("The accuracy of the CLASS 1 vs rest classifier on training data is", multiclass_accuracy(y, training_predictions)*100, "%")
#print(training_predictions)
for i in range(0):
  print(y, i, training_predictions_3vR[i][0], round(training_predictions_3vR[i][1]*100))
  i += 1
#Classifying Class 3 vs REST Test Data
'''
test_data_3vR = test_data
X_test = test_data_3vR[:, 0:-1]
y_test = test_data_3vR[:, -1]
'''
test_predictions_3vR = class_3vR_classifier.multiclass_predict(X_test)
#print("The accuracy of the CLASS 1 vs REST classifier on test data is", multiclass_accuracy(y_test, predictions)*100, "%")
#print(y_test)
#print(test_predictions_1vR)
##############################
#CLASS 3 VS REST END
##############################

for i in range(0):
  print(y_classlabels[i], training_predictions_1vR[i][0], round(training_predictions_1vR[i][1]*100), "||", training_predictions_2vR[i][0], round(training_predictions_2vR[i][1]*100), "||", training_predictions_3vR[i][0], round(training_predictions_3vR[i][1]*100))
  i += 1

ypred = np.copy(maxConfidence(training_predictions_1vR, training_predictions_2vR, training_predictions_3vR))
#print(y_classlabels)
#print(ypred)
print("The accuracy of the Multiclass Classifier on train data is", multiclass_accuracy(y_classlabels, ypred)*100, "%")

ypred = np.copy(maxConfidence(test_predictions_1vR, test_predictions_2vR, test_predictions_3vR))
#print(y_test)
#print(ypred)
print("The accuracy of the Multiclass Classifier on test data is", multiclass_accuracy(y_test, ypred)*100, "%")

print("\n")
##############################
#QUESTION 4 END
##############################