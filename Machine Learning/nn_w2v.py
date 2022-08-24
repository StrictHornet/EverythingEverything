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

  #print(tokenized_tweet)

Preprocess(train)
#print(train)
dataset = train.drop(['id'], axis=1)
dataset = dataset.dropna()
#print(dataset)
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 0].values
# print(X)
# print(y)
# for item in X:
#     print(item)
# print("XXXXXXXXXXXXX")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 77)
#print(x_train)
# vocab = []
# for each_list in x_train:
#     for word in each_list:
#         vocab.append(word)

#print(vocab)

n_dim = 300
epoch =5
#model = api.load('word2vec-google-news-300')
#Initialize model and build vocab
imdb_w2v = Word2Vec(vector_size=n_dim, min_count=5)

#Write function to remove words from tweets that fall below the minimum count

imdb_w2v.build_vocab(x_train)

#Train the model over train_reviews (this may take several minutes)
imdb_w2v.train(x_train, total_examples=imdb_w2v.corpus_count, epochs = epoch)

#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for any_word in text:
        try:
            vec += imdb_w2v.wv[any_word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    # print(vec)
    # print(text)
    # input("Press Enter to continue...")    
    return vec

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#for item in x_train:
train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
scaler.fit(train_vecs)
train_vecs = scaler.transform(train_vecs)

# test_vocab = []
# for each_list in x_test:
#     for tweet in each_list:
#         for word in tweet:
#             test_vocab.append(word)

#Train word2vec on test tweets
imdb_w2v.train(x_test, total_examples=imdb_w2v.corpus_count, epochs = epoch)

#Build test tweet vectors then scale
test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs = scaler.transform(test_vecs)

#Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
from sklearn.linear_model import SGDClassifier

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)

print('Test Accuracy: %.2f'%lr.score(test_vecs, y_test))
y_pred = lr.predict(test_vecs)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# from sklearn.model_selection import cross_val_score
# acc = cross_val_score(estimator = lr, X = x_train, y = y_train, cv = 10)
# print("Accuracy {:.2f}%".format(acc.mean()*100))
# print("Standard Deviation {:.2f} %".format(acc.std()*100))
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, pos_label=1))
print(f1_score(y_test, y_pred, pos_label=0))
print(f1_score(y_pred, y_test, pos_label=1))
print(f1_score(y_pred, y_test, pos_label=0))
shav = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
df = pd.DataFrame(shav)
df.to_csv('train.csv', index=False)
print(shav)

#FOR PUTTING VECTOR FILE IN-FILE
# from gensim.models import KeyedVectors
# from gensim import models

# word2vec_path = 'path/GoogleNews-vectors-negative300.bin.gz'
# w2v_model = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


from sklearn.svm import SVC
classifier = SVC(kernel = "linear", random_state = 10)
classifier.fit(train_vecs, y_train)
y_pred = classifier.predict(test_vecs)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator = classifier, X = train_vecs, y = y_train, cv = 10)
print("Accuracy {:.2f}%".format(acc.mean()*100))
print("Standard Deviation {:.2f} %".format(acc.std()*100))
print(f1_score(y_test, y_pred, pos_label=1))
print(f1_score(y_test, y_pred, pos_label=0))