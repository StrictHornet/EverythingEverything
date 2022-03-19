# Import libraries used in solution
import numpy as np
import pandas as pd
import warnings

#Import Training and Test Data
train_data = pd.read_csv('train.data', header=None)
test_data = pd.read_csv('test.data', header=None)

train_data = np.array(train_data.values)
test_data = np.array(test_data.values)

#Splitting Training Data
class1_train_data = train_data[0:40, :]
class2_train_data = train_data[40:80, :]
class3_train_data = train_data[80:, :]