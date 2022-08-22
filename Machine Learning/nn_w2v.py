import re
import pandas as pd 
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt 
import string
import nltk
# from nltk.stem.porter import *
# pstemmer = PorterStemmer()
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

from gensim.models import Word2Vec 
import gensim.downloader as api

train = pd.read_csv('train_dat.csv') #train
test = pd.read_csv('test_dat.csv')

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt 


def Preprocess(data):
  #Removing Handles
  data['tweet'] = np.vectorize(remove_pattern)(data['tweet'], "@[\w]*")
  #Removing Special Characters, Numbers, Punctuations
  data['tweet'] = data['tweet'].str.replace("[^a-zA-Z#]", " ")
  #Removing Shortwords
  data['tweet'] = data['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
  #Tokenizing tweets
  tokenized_tweet = data['tweet'].apply(lambda x: x.split())
  #Stemming
  tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
  data['tweet'] = tokenized_tweet

  print(tokenized_tweet)

Preprocess(train)
print(train)
dataset = train.drop(['id'], axis=1)
print(dataset)
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
print(X)
print(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#print(x_train)
vocab = []
for each_list in x_train:
    for tweet in each_list:
        for word in tweet:
            vocab.append(word)

#print(vocab)

n_dim = 300
epoch =15
#model = api.load('word2vec-google-news-300')
#Initialize model and build vocab
imdb_w2v = Word2Vec(vector_size=n_dim, min_count=4)
imdb_w2v.build_vocab(vocab)

#Train the model over train_reviews (this may take several minutes)
imdb_w2v.train(vocab, total_examples=imdb_w2v.corpus_count, epochs = epoch)

#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v.wv[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

from sklearn.preprocessing import scale
for item in x_train:
    train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in item])
train_vecs = scale(train_vecs)

test_vocab = []
for each_list in x_test:
    for tweet in each_list:
        for word in tweet:
            test_vocab.append(word)

#Train word2vec on test tweets
imdb_w2v.train(test_vocab, total_examples=imdb_w2v.corpus_count, epochs = epoch)

#Build test tweet vectors then scale
for item in x_test:
    test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in item])
test_vecs = scale(test_vecs)
print(test_vecs)

#Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
from sklearn.linear_model import SGDClassifier

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)

print('Test Accuracy: %.2f'%lr.score(test_vecs, y_test))

#FOR PUTTING VECTOR FILE IN-FILE
# from gensim.models import KeyedVectors
# from gensim import models

# word2vec_path = 'path/GoogleNews-vectors-negative300.bin.gz'
# w2v_model = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)