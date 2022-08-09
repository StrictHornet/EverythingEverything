import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
from nltk.stem.porter import *
warnings.filterwarnings("ignore", category=DeprecationWarning)

#%matplotlib inline

train = pd.read_csv("test_dat.csv")
test = pd.read_csv("train_dat.csv")
#print(train.head(3))
#
#Preprocessing tasks
#Handles, punctuations, short words, lemmings
# 

#Join two datasets to make preprocessing easier
big_data = train.append(test, ignore_index=True)

#Funtion to remove user id's from text using regx
def clean_id(input_text, element):
    found_elements =  re.findall(element, input_text)
    for user_id in found_elements:
        input_text = re.sub(user_id, "", input_text)

    return input_text

#Calling user id cleaner function
big_data['no-id'] = np.vectorize(clean_id)(big_data['tweet'], "@[\w]*")

#Calling cleaner function to clean special characters, punctuations and numbers
big_data['no-id'] = big_data['no-id'].str.replace("[^a-zA-Z#]"," ")

#Calling cleaner function to clean short words
big_data['no-id'] = big_data['no-id'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#print(big_data.head(3))

#Tokenization
tokenized_tweet = big_data['no-id'].apply(lambda x: x.split())

#Stemming of tokenized tweets
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) #stemming
#print(tokenized_tweet.head(5))

#Combine stemmed tweets
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

big_data['no-id'] = tokenized_tweet
print(big_data.head(30))