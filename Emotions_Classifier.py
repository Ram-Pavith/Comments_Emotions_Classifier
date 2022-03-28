import json
import pandas as pd
import numpy as np
import pickle
import sklearn as skl
import random


#Data Class
class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class Review:
	def __init__(self, text, score):
		self.text = text
		self.score = score
		self.sentiment = self.get_sentiment()
	def get_sentiment(self):
	    if self.score <= 2:
	    	return Sentiment.NEGATIVE
	    elif self.score == 3:
	    		return Sentiment.NEUTRAL
	    else: #Score of 4 or 5
	    	return Sentiment.POSITIVE


class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
        
    def get_text(self):
        return [x.text for x in self.reviews]
    
    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
        
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)

#LOAD DATA
file_name = "./data/sentiment/Books_small_10000.json"
reviews = []

with open(file_name) as f:
	for line in f:
		review = json.loads(line)
		reviews.append(Review(review['reviewText'], review['overall']))



#Prep Data
from sklearn.model_selection import train_test_split

training, test = train_test_split(reviews, test_size=0.33, random_state=42)
train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

#Bag of words vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# This book is great !
# This book was so bad
vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)


#Classification
#Different types of classifications

#Linear SVM
from sklearn import svm
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)
clf_svm.predict(test_x_vectors[0])

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)
clf_dec.predict(test_x_vectors[0])

#Naive Bayes

from sklearn.naive_bayes import GaussianNB
clf_gnb = DecisionTreeClassifier()
clf_gnb.fit(train_x_vectors, train_y)
clf_gnb.predict(test_x_vectors[0])

#Logistic Regression
from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)
clf_log.predict(test_x_vectors[0])



#Tuning our model (with Grid Search)
from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf'), 'C': (1,4,8,16,32)}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_x_vectors, train_y)


#Saving Model
import pickle

with open('./models/sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('./models/entiment_classifier.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)
loaded_clf.predict(test_x_vectors[0])

from sklearn.linear_model import Perceptron

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)  