import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

#%matplotlib inline

train = pd.read_csv("test_dat.csv")
test = pd.read_csv("train_dat.csv")

print(train.head(3))#

#
#Preprocessing tasks
#Handles, punctuations, short words, lemmings
# 

