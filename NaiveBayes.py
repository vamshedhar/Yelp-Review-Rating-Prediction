from __future__ import print_function

from collections import Counter
import numpy as np
import math
import csv

from pyspark import SparkContext

class NaiveBayes:

    ''' function to initialize all the class variables '''
	def __init__(self, filepath, classes):
		self.class_probability = {}
		self.filepath = filepath
		self.classes = classes
		self.data = []

        ''' get an existing spark session or create a new session '''
		self.sc = SparkContext.getOrCreate()

    ''' function to replace ratings from 1-5 to 1-3 for modelling 3-class problem '''
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

    ''' funtion to load data from input csv file '''
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

    ''' function to check if predicted rating matches with actual rating '''
	def validate_class(self, prediction):
		return prediction[0] == prediction[1]

    ''' funtion to calculate accuracy of built model by taking fraction of reviews that are 
    correctly classified among all the testing reviews '''    
	def calculate_accuracy(self, predictions):
		return sum(predictions) * 100 / float(len(predictions))

	''' function to train the model with training subset of data '''
    def run_model(self):

		self.load_data()

		print("Running Naive Bayes model")
		review_data = self.sc.parallelize(self.data)

		training_data, test_data = review_data.randomSplit([0.8, 0.2])

        ''' splitting all the reviews with space as delimiter to form vocabulary of training data '''
		words_in_reviews = training_data.map(lambda x: x[0].split(' '))
		words_in_training = words_in_reviews.flatMap(lambda y : y).collect()
		word_counter_training = Counter(words_in_training)
		vocabulary_count_training = len(word_counter_training)

		review_count_training = training_data.count()

		rating_review_list = training_data.map(lambda x: (x[1], x[0]))

		class_vocab_count = []
		class_vocab_counter = []

        ''' extracting vocabulary and words count for each class '''
		for classIndex in range(self.classes):
			words_reviews_in_class = self.sc.parallelize(rating_review_list.lookup(classIndex + 1)).map(lambda x: x.split(' '))
			words_in_class = words_reviews_in_class.flatMap(lambda y : y).collect()

			words_in_class_counter = Counter(words_in_class)

			class_vocab_count.append(len(words_in_class_counter))
			class_vocab_counter.append(words_in_class_counter)

			self.class_probability[classIndex + 1] = len(rating_review_list.lookup(classIndex + 1))/float(review_count_training)

        ''' assuming learning rate as 1'''
		alpha = 1.0

		while(alpha <= 1):
			print("Running Model for alpha = " + str(alpha))
			alpha_vocab_length = alpha * vocabulary_count_training

			self.class_vocab_probability = []

            ''' calculating conditional probabilities for all the words in each class and forming a dictionary 
                with word and its count in that class'''
			for classIndex in range(self.classes):
				vocab_counter = self.sc.parallelize(class_vocab_counter[classIndex].items())
				vocab_probability = vocab_counter.map(lambda x: (x[0], np.divide(np.add(x[1], alpha), np.add(class_vocab_count[classIndex], alpha_vocab_length))))
				self.class_vocab_probability.append(dict(vocab_probability.collect()))

			classes = self.classes
			class_probability = self.class_probability
			class_vocab_probability = self.class_vocab_probability
            
            ''' predicting ratings for all the reviews in testing data '''
			test_data_vocab = test_data.map(lambda x: [x[1]] + x[0].split(' '))
			validated_test_data = test_data_vocab.map(lambda x: predict_class(x, class_vocab_probability, classes, class_probability)).map(lambda x: x[0] == x[1])
            
            
            ''' calculating accuracy for the predicted ratings '''
			accuracy = self.calculate_accuracy(validated_test_data.collect())

			print("\n\n")
			print("Accuracy with alpha {0} for NaiveBayes is {1}".format(str(alpha), str(accuracy)))
			print("\n\n")
			alpha += 1

		# self.sc.close()

    ''' function to preprocess input review before performing prediction '''
	def predict_rating(self, review):
		words = [0] + review.split(' ')
		prediction = predict_class(words, self.class_vocab_probability, self.classes, self.class_probability)
		return prediction[1]

''' function to predict the rating for a given review using the trained model '''
def predict_class(words, class_vocab_probability, classes, class_probability):
	actual_class = words[0]

	words = words[1:]
	
	predicted_probability = []

    ''' initializing predicted probability to the class probabilities '''
	for classIndex in range(classes):
		predicted_probability.append(math.log10(class_probability[classIndex + 1]))

    ''' updating predicted probability by adding conditional probabilities of each word in the given review  '''
	for i in range(len(words)):
		for classIndex in range(classes):
			if words[i] in class_vocab_probability[classIndex]:
				predicted_probability[classIndex] = predicted_probability[classIndex] + math.log10(class_vocab_probability[classIndex][words[i]])

    ''' updating final rating as the class corresponding to highest predicted probability '''
	max_probability = max(predicted_probability)
	predicted_class = predicted_probability.index(max_probability) + 1

	return (actual_class, predicted_class)

if __name__ == "__main__":

	model = NaiveBayes("stemmed_reviews.csv", 5)

	model.run_model()

	review = ""
	while (review.lower() != "quit"):
		review = input("Please enter a review to predict: ")
		rating = model.predict_rating(str(review))

		print("Rating: " + str(rating))


