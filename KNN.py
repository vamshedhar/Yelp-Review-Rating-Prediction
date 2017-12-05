from __future__ import print_function
import csv

from collections import Counter

from pyspark import SparkContext

from operator import itemgetter

import math
import nltk


def get_clean_review(raw_review):
	PorterStemmer = nltk.stem.PorterStemmer()
	letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
	words = letters_only.lower().split()
	stops = load_stopwords()
	meaningful_words = [PorterStemmer.stem(w) for w in words if not w in stops]
	return( " ".join( meaningful_words ))


'''Function to calculate votes for neighbours and return predicted rating'''
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=itemgetter(1), reverse=True)
	return sortedVotes[0][0]

'''Function to calculate hamming distance between two strings'''
def hamdist(word1, word2):
	number_differences = 0

	word1 = word1[0]
	word2 = word2[0]

	if len(word1) != len(word2):
		return max(len(word1), len(word2))
	for char1, char2 in zip(word1, word2):
		if char1 != char2:
			number_differences += 1
	return number_differences


class KNN:

	def __init__(self, filepath, classes):
		self.filepath = filepath
		self.classes = classes
		self.data = []

		self.sc = SparkContext.getOrCreate()
	'''Function to convert 5 class labels to 3 class labels if 3 class classification is chosen'''
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
				self.data.append([row[0], self.convert_rating(row[1])])
				count = count + 1
				if count == int(total_lines * percent):
					print( str(math.ceil(percent * 100)) + "% complete")
					percent += 0.1

	''' Function to calculate accurancy of KNN model built'''
	def getAccuracyKNN(self, testSet, predictions):
		correct = 0
		for x in range(len(testSet)):
			if testSet[x][-1] == predictions[x]:
				correct += 1
		return (correct / float(len(testSet))) * 100.0

	''' Function to run KNN model'''
	def run_model(self):

		print("Running KNN model")

		self.load_data()

		review_data = self.sc.parallelize(self.data)

		training_data, test_data = review_data.randomSplit([0.999, 0.001])

		k = int(math.sqrt(len(self.data))) - 15

		kMax = k + 30

		hamming_distances_RDD = test_data.cartesian(training_data).map(lambda x: (x[0], [hamdist(x[0], x[1])] + x[1]))
		hamming_distances_sorted_RDD = hamming_distances_RDD.sortBy(lambda x: int(x[1][0])).map(lambda x: (tuple(x[0]), x[1]))
		hamming_distances_grouped_RDD = hamming_distances_sorted_RDD.groupByKey()

		nearest_neighbours_RDD = hamming_distances_grouped_RDD.map(lambda x: (x[0], list(x[1])))

		while (k <= kMax):

			k_nearest_neighbours_RDD = nearest_neighbours_RDD.map(lambda x: (x[0], x[1][0:k]))
			predicted_data = k_nearest_neighbours_RDD.map(lambda x: (x[0], getResponse(x[1])))

			test_reviews = predicted_data.keys().collect()
			predicted_ratings = predicted_data.values().collect()

			accuracy = self.getAccuracyKNN(test_reviews, predicted_ratings)
			print("#################")
			print('Accuracy: for KNN with k '+ repr(k)+ ' is ' + repr(accuracy) + '%')
			print("#################")

			k=k+5

	'''Function to predict rating for a given review'''
	def predict_rating(self, review):
		review_data = self.sc.parallelize(self.data)
		test_data = self.sc.parallelize([review])

		k = int(math.sqrt(len(self.data))) - 15

		hamming_distances_RDD = test_data.cartesian(review_data).map(lambda x: (x[0], [hamdist(x[0], x[1])] + x[1]))
		hamming_distances_sorted_RDD = hamming_distances_RDD.sortBy(lambda x: int(x[1][0])).map(lambda x: (tuple(x[0]), x[1]))

		hamming_distances_grouped_RDD = hamming_distances_sorted_RDD.groupByKey()

		nearest_neighbours_RDD = hamming_distances_grouped_RDD.map(lambda x: (x[0], list(x[1])))

		k_nearest_neighbours_RDD = nearest_neighbours_RDD.map(lambda x: (x[0], x[1][0:k]))
		predicted_data = k_nearest_neighbours_RDD.map(lambda x: (x[0], getResponse(x[1])))

		predicted_ratings = predicted_data.values().collect()

		return predicted_ratings[0]


if __name__ == "__main__":

	'''Create K Nearest Neighbours model and run'''
	model = KNN("reviews.csv", 5)

	model.run_model()

	review = ""
	while (True):
		review = input("Please enter a review to predict: ")
		if review.lower() == "quit":
			break
		if review != "":
			clean_review = get_clean_review(str(review))
			rating = model.predict_rating(clean_review)

			print("Rating: " + str(rating))




