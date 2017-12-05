from __future__ import print_function
import csv

from collections import Counter

from pyspark import SparkContext

from operator import itemgetter

import math


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

def hamdist(str1, str2):
	diffs = 0

	str1 = str1[0]
	str2 = str2[0]

	if len(str1) != len(str2):
		return max(len(str1), len(str2))
	for ch1, ch2 in zip(str1, str2):
		if ch1 != ch2:
			diffs += 1
	return diffs


class KNN:

	def __init__(self, filepath, classes):
		self.filepath = filepath
		self.classes = classes
		self.data = []

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
				self.data.append([row[0], self.convert_rating(row[1])])
				count = count + 1
				if count == int(total_lines * percent):
					print( str(math.ceil(percent * 100)) + "% complete")
					percent += 0.1


	def getAccuracyKNN(self, testSet, predictions):
		correct = 0
		for x in range(len(testSet)):
			if testSet[x][-1] == predictions[x]:
				correct += 1
		return (correct / float(len(testSet))) * 100.0


	def run_model(self):

		print("Running KNN model")

		self.load_data()

		review_data = self.sc.parallelize(self.data)

		training_data, test_data = review_data.randomSplit([0.999, 0.001])

		k = int(math.sqrt(len(self.data))) - 15

		kMax = k + 30

		nearestNeigboursHAM = test_data.cartesian(training_data).map(lambda x: (x[0], [hamdist(x[0], x[1])] + x[1]))
		nearestNeigboursSORT = nearestNeigboursHAM.sortBy(lambda x: int(x[1][0])).map(lambda x: (tuple(x[0]), x[1]))

		nearestNeigboursGROUP = nearestNeigboursSORT.groupByKey()

		nearestNeigbours = nearestNeigboursGROUP.map(lambda x: (x[0], list(x[1])))

		while (k <= kMax):

			nearestNeigbours2 = nearestNeigbours.map(lambda x: (x[0], x[1][0:k]))
			nearestNeigbours3 = nearestNeigbours2.map(lambda x: (x[0], getResponse(x[1])))

			testsetfinal = nearestNeigbours3.keys().collect()
			predicted = nearestNeigbours3.values().collect()

			accuracy = self.getAccuracyKNN(testsetfinal, predicted)
			print("#################")
			print('Accuracy: for KNN with k '+ repr(k)+ ' is ' + repr(accuracy) + '%')
			print("#################")

			k=k+5

	def predict_rating(self, review):
		review_data = self.sc.parallelize(self.data)
		test_data = self.sc.parallelize([review])

		k = int(math.sqrt(len(self.data))) - 15

		nearestNeigboursHAM = test_data.cartesian(review_data).map(lambda x: (x[0], [hamdist(x[0], x[1])] + x[1]))
		nearestNeigboursSORT = nearestNeigboursHAM.sortBy(lambda x: int(x[1][0])).map(lambda x: (tuple(x[0]), x[1]))

		nearestNeigboursGROUP = nearestNeigboursSORT.groupByKey()

		nearestNeigbours = nearestNeigboursGROUP.map(lambda x: (x[0], list(x[1])))

		nearestNeigbours2 = nearestNeigbours.map(lambda x: (x[0], x[1][0:k]))
		nearestNeigbours3 = nearestNeigbours2.map(lambda x: (x[0], getResponse(x[1])))

		predicted = nearestNeigbours3.values().collect()

		return predicted[0]


if __name__ == "__main__":

	model = KNN("reviews.csv", 5)

	model.run_model()

	review = ""
	while (True):
		review = input("Please enter a review to predict: ")
		if review.lower() == "quit":
			break
		if review != "":	
			rating = model.predict_rating(str(review))

			print("Rating: " + str(rating))



