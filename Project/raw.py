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

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

train['tidy_tweet'] = tokenized_tweet

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(train['tidy_tweet'])
#vocab = bow.get_feature_names()

newb = bow.toarray()

print(newb)
print(newb.shape)


#designed to see output format of bow model