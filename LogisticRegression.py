from __future__ import print_function
import csv
import sys

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
import re
from nltk.corpus import stopwords


def load_stopwords():
	return set(stopwords.words("english"))

def get_clean_review(raw_review):
	PorterStemmer = nltk.stem.PorterStemmer()
	letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
	words = letters_only.lower().split()
	stops = load_stopwords()
	meaningful_words = [PorterStemmer.stem(w) for w in words if not w in stops]
	return( " ".join( meaningful_words ))

def sigmoid(z):
		return 1 / (1 + np.exp(-z)) 
'''Function to check y-label and return 1 for class i and 0 for other classes'''
def check_rating(x, i):
	if(x[0]==i):
		x[0]=1
	else:
		x[0]=0
	return x

'''Function to calculate gradient'''
def calculate_gradient(input_vector, beta):
	rating_current_review = input_vector[0]
	input_vector[0] = 1.0
	X = np.asmatrix(input_vector, dtype=float).T
	scores = np.dot(beta.T,X)
	residue = rating_current_review - sigmoid(scores)
	gradient = np.dot(X,residue)
	return gradient

'''Function to pedict class for test input using beta vectors calculated with training data'''
def predict_rating(test_review, beta_all):
	x = np.asarray(test_review)
	x = np.append([1], x)
	x = x.reshape(1, x.shape[0])
	h_beta_x = [(np.dot(x, beta.T), class_label) for beta, class_label in beta_all]
	return max(h_beta_x, key=itemgetter(0))[1]

'''Function to convert matrix to array'''
def reduceMatrix(input_matrix):
	converted_array = np.asarray(input_matrix)
	return converted_array[0]

class LogisticRegression:

	'''Dunction to initialize all class level variables'''
	def __init__(self, filepath, classes):
		self.filepath = filepath
		self.classes = classes
		self.reviews = []
		self.stars = []

		self.sc = SparkContext.getOrCreate()

	'''Function to convert 5 class labels to 3 class labels if 3 class classification is chosen'''
	def convert_rating(self, rating):
		intRating = int(rating[0])

		if self.classes == 5:
			return intRating
		else:
			if intRating <= 2:
				return 1
			elif intRating == 3:
				return 2
			else:
				return 3

	''' Function to load data from csv file and prepare input data structures'''
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

	'''Function to run Logistic Regression model'''
	def run_model(self):

		self.load_data()

		'''Calculate TF-IDF values removing English stop words and considering bi-gram'''
		tfidf_vectorizer = TfidfVectorizer(max_df=0.90, max_features=500, stop_words='english', use_idf=True, tokenizer=nltk.word_tokenize, ngram_range=(2,2))

		tfidf_data = tfidf_vectorizer.fit_transform(self.reviews)

		X = tfidf_data
		y = np.asarray(self.stars).astype(float)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

		y_matrix = y_train.reshape(X_train.shape[0], 1)

		training_data = sp.sparse.hstack((y_matrix, X_train))

		training_data.tocsr()

		data_parallelized = self.sc.parallelize(training_data.tocsr().todense())

		beta_all = []

		'''Calculating Beta values for each class'''
		for i in np.unique(y):
			beta = np.zeros(X_train.shape[1] + 1).reshape(X_train.shape[1] + 1, 1)  
			data_transformed = data_parallelized.map(lambda x: reduceMatrix(x))
			Xy_two_class = data_transformed.map(lambda x: check_rating(x,i))

			number_iterations = 10
			alpha = 0.001
			
			print("class " + str(i))

			'''Calculate beta values for a class'''
			for j in range(number_iterations):
				print("iteration " + str(j))
				gradient=Xy_two_class.map(lambda x: calculate_gradient(x, beta))
				gradient = gradient.reduce(lambda a,b:a+b)
				beta=beta+alpha*gradient

			beta_all.append((beta.T,i))

		test_data = self.sc.parallelize(X_test.todense())

		self.beta_all = beta_all

		y_predicted = test_data.map(lambda x: predict_rating(x, beta_all)).collect()
		y_predicted = np.asarray(y_predicted)

		'''Calculating accuracy using model built'''
		print('Accuracy from scratch: {0}'.format((y_predicted == y_test.T).sum().astype(float) / len(y_test)))

		'''Calculating accuracy using scikit learn libraries'''
		clf = SKLogisticRegression(fit_intercept=True, C = 1e15)
		clf.fit(X_train, y_train)
		print('Accuracy from sk-learn: {0}'.format(clf.score(X_test, y_test)))

	'''Predicting rating for a test review '''
	def predict_new_review_rating(self, review):
		new_reviews = self.reviews + [review]
		tfidf_vectorizer = TfidfVectorizer(max_df=0.90, max_features=500, stop_words='english', use_idf=True, tokenizer=nltk.word_tokenize, ngram_range=(2,2))
		tfidf_data = tfidf_vectorizer.fit_transform(new_reviews)
		f = tfidf_data.todense()[tfidf_data.shape[0] - 1]
		return predict_rating(f, self.beta_all)



if __name__ == "__main__":

	if len(sys.argv) != 3:
		print("Please give valid inputs arguments!")
		print("python Model.py <inputfile> <classcount>")
		sys.exit()

	filepath = str(sys.argv[1])

	classes = int(sys.argv[2])

	if classes !=5 and classes != 3:
		print("Class count can either be 3 or 5")
		sys.exit()

	'''Create Logistic Regression model and run'''
	model = LogisticRegression(filepath, classes)

	model.run_model()

	review = ""
	while (True):
		review = input("Please enter a review to predict: ")
		if review.lower() == "quit":
			break
		if review != "":
			clean_review = get_clean_review(str(review))
			rating = model.predict_new_review_rating(clean_review)

			print("Rating: " + str(rating))

