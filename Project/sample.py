import re
import pandas as pd 
import numpy as np  
import seaborn as sns
import string
import nltk
from nltk.stem.porter import *

import gensim.downloader as glove_api

train  = pd.read_csv('train_dat.csv')

print(train.head())



def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt 


print('\n\nRemoving  Twitter Handles \n\n')
train['tidy_tweet'] = np.vectorize(remove_pattern)(train['tweet'], "@[\w]*")

print(train['tidy_tweet'].head())


print('\n\nRemoving Short Words\n\n')

train['tidy_tweet'] = train['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")

train['tidy_tweet'] = train['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

print(train['tidy_tweet'].head())


print('\n\nTweet Tokenization\n\n')
tokenized_tweet = train['tidy_tweet'].apply(lambda x: x.split())
print('WECSSSSSSSSSSSS')
print(tokenized_tweet.head())


print('\n\nStemming\n\n')

stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
 # stemming
print(tokenized_tweet.head())
print("WD")
print(tokenized_tweet[0][1])


#### Word embedding
glove_model = glove_api.load('glove-twitter-25')
sample_glove_embedding = glove_model[tokenized_tweet[3][1]]
print(sample_glove_embedding)
print(glove_model['love'])

old_data = tokenized_tweet[0:5]
old_data_label = train.iloc[0:5, 1:2]
print(old_data)

new_data = []
for item in old_data:
    sentence_no = []
    for word in item:
        try:
            word_no = glove_model[word]
        except KeyError:
            continue
        sentence_no.append(word_no)

    new_data.append(sentence_no)

#print(new_data)
#data = pd.DataFrame(new_data)
csvdata = np.asarray(new_data)
print(csvdata)
print(type(csvdata))
#print(csvdata.shape)

data = pd.DataFrame(csvdata)

import csv
  
data.to_csv('data.csv', index=False)
print(old_data_label)

existing_label = np.concatenate([csvdata, old_data_label], axis = 1)
print(existing_label)
###
#Log Reg Model
###

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = old_data
# test_bow = bow[31962:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,0] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

good_job = f1_score(yvalid, prediction_int) # calculating f1 score
print(good_job)

#Stopped at converting glove vectors to numpy array to allow it be formatted as input data for logistic regression model