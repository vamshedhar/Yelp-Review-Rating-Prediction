# PREDICTING​ ​YELP​ ​FOOD​ ​REVIEWS’​ ​RATINGS

## Table of Contents
 - [Introduction](#introduction)
 - [Dataset](#dataset)
    - [Data Collection](#data-collection)
    - [Data Processing](#data-processing)
 - [Algorithms](#algorithms)
    - [Naive Bayes](#naive-bayes)
    - [K- Nearest Neighbours](#k-nearest-neighbours)
	- [Logistic Regression](#logistic-regression)
 - [Results and Analysis](#results-and-analysis)
    - [Naive Bayes Output](#naive-bayes-output)
    - [K- Nearest Neighbours Output](#k-nearest-neighbours-output)
	- [Logistic Regression Output](#logistic-regression-output)
 - [Observations](#observations)
 - [Future Enhancements](#future-enhancements)
 - [References](#references)
 
 

## Algorithms
One of the categories that our project could fall under is definitely the supervised
learning. The main reason for this is that a prior knowledge of the target variable is already known, that needs to be predicted at the end of the project. To train our prediction models, we used three supervised learning algorithms.

### Naive Bayes
Logistic regression is a simple classification algorithm for learning to predict a discrete variable such as predicting whether a grid of pixel intensities represents a “0” digit or a “1” digit. Here we use a hypothesis class to try to predict the probability that a given sample belongs to the class “1” versus the probability that it belongs to the class “0”. Specifically, we will try to learn a function of the form:

	<center>P(y=1|x) = hθ(x) = 1/1+exp(-θTx)</center>
	<center>P(y=0|x) = 1- P(y=1|x)</center>


### K- Nearest Neighbours
### Logistic Regression

## Results and Analysis
We built models for both 3-class and 5-class classification problem using above 3
supervised learning models. For 3-class classification, we programmatically constructed Good, Average and Bad classes from review labels.

### Naive Bayes Output

### K- Nearest Neighbours Output
### Logistic Regression Output


## Observations
 - KNN performed better when k value is chosen around square root of N(number of records in training data)
 - Built models performed better when bi-gram is chosen
 - Degree of positivity or negativity was observed to be very important for 5-class
classification problem. However, lemmatization of words couldn’t preserve this
information( lemma of word “worst”(class-1) is “bad”(class-2))
 - Because of conditional independence assumption of Naive Bayes, model failed to capture the difference between the phrases “good” and “not good”


## Future Enhancements
 - Refine KNN model by using different similarity measurements like cosine and minkowski distance
 - Perform better feature extraction and reduce dimensionality for improved performance

## References

1. https://pdfs.semanticscholar.org/ce23/988aa1830c0b343e64234f318f28b91108d3.pdf
2. https://arxiv.org/pdf/1605.05362.pdf
3. https://cseweb.ucsd.edu/~jmcauley/cse255/reports/fa15/017.pdf
4. https://pdfs.semanticscholar.org/ce23/988aa1830c0b343e64234f318f28b91108d3.pdf
5. http://cs229.stanford.edu/proj2011/MehtaPhilipScariaPredicting%20Star%20Ratings%20from%20Movie%20Review%20Comments.pdf

