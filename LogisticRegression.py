from __future__ import print_function
import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from pyspark import SparkContext
import nltk
import numpy as np
import scipy as sp
import math
from operator import itemgetter

from sklearn.linear_model import LogisticRegression as SKLogisticRegression

def sigmoid(z):
		return 1 / (1 + np.exp(-z)) 

def check_ylabel(x, i):
	if(x[0]==i):
		x[0]=1
	else:
		x[0]=0
	return x

def gradiant_fun(inputVector, beta):
	y_vec = inputVector[0]
	inputVector[0] = 1.0
	X = np.asmatrix(inputVector, dtype=float).T
	scores = np.dot(beta.T,X)
	residue = y_vec - sigmoid(scores)
	gradiant = np.dot(X,residue)
	return gradiant

def predict(x_test, beta_all):
	x = np.asarray(x_test)
	x = np.append([1], x)
	x = x.reshape(1, x.shape[0])
	values = [(np.dot(x, w.T), class_label) for w, class_label in beta_all]
	return max(values, key=itemgetter(0))[1]

def reduceMatrix(x):
	a = np.asarray(x)
	return a[0]

class LogisticRegression:

	def __init__(self, filepath, classes):
		self.filepath = filepath
		self.classes = classes
		self.reviews = []
		self.stars = []

		self.sc = SparkContext.getOrCreate()

	def convert_rating(self, rating):
		intRating = int(rating[0])

		if self.classes == 5:
			return intRating
		else:
			if intRating <= 2:
				return 0
			elif intRating == 3:
				return 1
			else:
				return 2


	def load_data(self):
		print ("Loading data fron File:")
		total_lines = sum([1 for row in open(self.filepath, 'rU')]) - 1
		count = 0
		percent = 0.1
		with open(self.filepath, 'rU') as f:
			reader = csv.reader(f)
			next(reader, None)
			for row in reader:
				self.reviews.append(row[0])
				self.stars.append(self.convert_rating(row[1]))
				count = count + 1
				if count == int(total_lines * percent):
					print( str(math.ceil(percent * 100)) + "% complete")
					percent += 0.1

	def run_model(self):

		self.load_data()

		tfidf_vectorizer = TfidfVectorizer(max_df=0.90, max_features=500, stop_words='english', use_idf=True, tokenizer=nltk.word_tokenize, ngram_range=(2,2))

		tfidf_data = tfidf_vectorizer.fit_transform(self.reviews)

		X = tfidf_data
		y = np.asarray(self.stars).astype(float)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

		y_matrix = y_train.reshape(X_train.shape[0], 1)

		X_y = sp.sparse.hstack((y_matrix, X_train))

		X_y.tocsr()

		Xy = self.sc.parallelize(X_y.tocsr().todense())

		beta_all = []

		for i in np.unique(y):
			beta = np.zeros(X_train.shape[1] + 1).reshape(X_train.shape[1] + 1, 1)  
			Xy2 = Xy.map(lambda x: reduceMatrix(x))
			Xy_new = Xy2.map(lambda x: check_ylabel(x,i))

			num_iter = 10
			alpha = 0.001
			
			print("class " + str(i))

			for j in range(num_iter):
				print("iteration " + str(j))
				gradiant=Xy_new.map(lambda x: gradiant_fun(x, beta))
				gradiant = gradiant.reduce(lambda a,b:a+b)
				beta=beta+alpha*gradiant

			beta_all.append((beta.T,i))

		X_test2 = self.sc.parallelize(X_test.todense())

		self.beta_all = beta_all

		y_pred = X_test2.map(lambda x: predict(x, beta_all)).collect()
		y_pred = np.asarray(y_pred)

		#calculating accuracy   
		print('Accuracy from scratch: {0}'.format((y_pred == y_test.T).sum().astype(float) / len(y_test)))

		clf = SKLogisticRegression(fit_intercept=True, C = 1e15)
		clf.fit(X_train, y_train)
		print('Accuracy from sk-learn: {0}'.format(clf.score(X_test, y_test)))

	def predict_rating(self, review):
		new_reviews = self.reviews + ["amazing food great service"]
		tfidf_vectorizer = TfidfVectorizer(max_df=0.90, max_features=500, stop_words='english', use_idf=True, tokenizer=nltk.word_tokenize, ngram_range=(2,2))
		tfidf_data = tfidf_vectorizer.fit_transform(new_reviews)
		f = tfidf_data.todense()[tfidf_data.shape[0] - 1]
		return predict(f, self.beta_all)



if __name__ == "__main__":

	model = LogisticRegression("reviews.csv", 5)

	model.run_model()

	review = ""
	while (review.lower() != "quit"):
		review = input("Please enter a review to predict: ")
		rating = model.predict_rating(str(review))

		print("Rating: " + str(rating))

