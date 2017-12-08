# PREDICTING YELP FOOD REVIEWS’ RATINGS

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
 
## Introduction
The growth of the World Wide Web has resulted in troves of reviews for products we wish to purchase, destinations we may want to travel to, and decisions we make on a day to day basis. Using machine learning techniques to infer the polarity of text-based comments is of great importance in this age of information. While there have been many successful attempts at binary sentiment analysis of text-based content, fewer attempts have been made to classify texts into more granular sentiment classes. The aim of this project is to predict the star rating of
a user’s comment about a restaurant on a scale of 1, 2, 3 …to 5. Though a user self-classifies a comment on a scale of 1 to 5 in our dataset from Yelp, this research can potentially be applied to settings where a comment about a restaurant is available, but no corresponding star rating is (i.e. the sentiment expressed in review about a restaurant).
 
## Dataset
### Data collection
The obtained data-set is in the JSON format containing 6 million records. Some of the key statistics of [Yelp dataset](https://www.yelp.com/dataset/challenge) are as follows:

	Businesses       - 156,639
	Check-ins        - 135,148
	Users            - 1,183,362
	Tip              - 1,028,802
	Reviews          - 4,736,897
### Data Processing
Yelp consists of several categories of businesses like Restaurants, Hotels, Home Service etc., of which restaurants contributed to more than 60%. As the part of the initial analysis, we considered Restaurant business and analyzed reviews of this business across multiple cities. We only used businesses and reviews data from the dataset.

#### Sample business object

    {
	    // string, 22 character unique string business id
	    "business_id": "tnhfDv5Il8EaGSXZGiuQGg",

	    // string, the business's name
	    "name": "Garaje",

	    // string, the neighborhood's name
	    "neighborhood": "SoMa",

	    // string, the full address of the business
	    "address": "475 3rd St",

	    // string, the city
	    "city": "San Francisco",

	    // string, 2 character state code, if applicable
	    "state": "CA",

	    // string, the postal code
	    "postal code": "94107",

	    // float, latitude
	    "latitude": 37.7817529521,

	    // float, longitude
	    "longitude": -122.39612197,

	    // float, star rating, rounded to half-stars
	    "stars": 4.5,

	    // interger, number of reviews
	    "review_count": 1198,

	    // integer, 0 or 1 for closed or open, respectively
	    "is_open": 1,

	    // object, business attributes to values. note: some attribute values might be objects
	    "attributes": {
	        "RestaurantsTakeOut": true,
	        "BusinessParking": {
	            "garage": false,
	            "street": true,
	            "validated": false,
	            "lot": false,
	            "valet": false
	        },
	    },

	    // an array of strings of business categories
	    "categories": [
	        "Mexican",
	        "Burgers",
	        "Gastropubs"
	    ],

	    // an object of key day to value hours, hours are using a 24hr clock
	    "hours": {
	        "Monday": "10:00-21:00",
	        "Tuesday": "10:00-21:00",
	        "Friday": "10:00-21:00",
	        "Wednesday": "10:00-21:00",
	        "Thursday": "10:00-21:00",
	        "Sunday": "11:00-18:00",
	        "Saturday": "10:00-21:00"
	    }
    }

#### Sample review object

	{
	    // string, 22 character unique review id
	    "review_id": "zdSx_SD6obEhz9VrW9uAWA",

	    // string, 22 character unique user id, maps to the user in user.json
	    "user_id": "Ha3iJu77CxlrFm-vQRs_8g",

	    // string, 22 character business id, maps to business in business.json
	    "business_id": "tnhfDv5Il8EaGSXZGiuQGg",

	    // integer, star rating
	    "stars": 4,

	    // string, date formatted YYYY-MM-DD
	    "date": "2016-03-09",

	    // string, the review itself
	    "text": "Great place to hang out after work: the prices are decent, and the ambience is fun. It's a bit loud, but very lively. The staff is friendly, and the food is good. They have a good selection of drinks.",

	    // integer, number of useful votes received
	    "useful": 0,

	    // integer, number of funny votes received
	    "funny": 0,

	    // integer, number of cool votes received
	    "cool": 0
	}

Firstly, the data set was in JSON format, we used some python code to convert it to CSV files. We used only `business_id`, `categories` and `city` from the business object and `review_id`, `business_id`, `text` and `stars` from the review object. Then we used [pandas](https://pandas.pydata.org/) library to perform analysis and operations on CSV file. We first joined the businesses data and the reviews data and filtered out the reviews for category restaurants.  

Total review count for restaurant category came out to be 2,927,731. Then we performed analysis on city wise review count. The top five cities based on review count is as follows:

	Las Vegas	-	849,883
	Phoenix		-	302,403
	Toronto		-	276,887
	Scottsdale	-	164,893
	Charlotte	-	141,281

We divided the data based on cities in order to perform algorithms on whole data set and part of the dataset. From the business and review combined data, we selected only review text and start rating as our main goal is to analyze text and predict the review. We created CSV files for individual cities to perform further processing.

As the part of next step - dataset preprocessing, actions taken were

 - Cleaning and Eliminating foreign language reviews
 - Removing stop words
 - Stemming
 - Lemmatization
 - Bi-gram technique

For eliminating stop words, we used nltk stopwords library. For stemming, we have made use of the porter’s stemming algorithm to continue with the process of stemming the words.This process helped to get to the root form of every word which simplifies the classification process as the stemmer reduces a particular word to its dictionary form and can, therefore, represent most words in the data- set and avoid redundancy. We also tried to do lemmatization, but lemmatization on words “worst” and “bad” yielded “bad”. In our 5 class classification, we needed
to restore the degree of negativity in review text to correctly classify reviews. As the part of next step, we considered uni-gram, bi-gram(because “good” and “not good” holds totally different meaning in our context) and the combination of both to extract features from review text and evaluated accuracies with each of these techniques.

## Algorithms
One of the categories that our project could fall under is definitely the supervised
learning. The main reason for this is that a prior knowledge of the target variable is already known, that needs to be predicted at the end of the project. To train our prediction models, we used three supervised learning algorithms.

### Naive Bayes
Naive Bayes (Ng and Jordan, 2002) classifier makes the Naive Bayes assumption (i.e. it assumes conditional independence between any pair of features given some class) to model the joint probability P(r, s) for any feature vector r and star rating s. Then, given a new feature vector r<sup>∗</sup> for a new review r<sup>∗</sup>, the joint probability function is computed for all values of s, and the s value corresponding to the highest probability is output as the final class label for review r<sup>∗</sup>.

The multinomial Naive Bayes(which assumes that P(r<sub>i</sub>|s) is a multinomial distribution for all i) with smoothing has been implemented. To classify, we calculated probabilities of the review belonging to each rating and then selected the class value with the highest probability. We have performed computations by summing logs of probabilities rather than multiplying probabilities for underflow prevention. We ran this model for smoothing factor - alpha ranging from 1 to 5 and found that we had better results for alpha = 1; so we’re considering alpha as 1 for our model.

<img src="https://raw.githubusercontent.com/vamshedhar/YelpReviewImages/master/NB%20Formula.png" alt="Naive Bayes Formula" width="400" />

### K-Nearest Neighbours
K nearest neighbors is a supervised classification algorithm that uses all available data samples which classifies a new sample based on a similarity measure. There are many such similarity measures namely: Euclidean Distance, Hamming Distance, Cosine Similarity etc. Since we are working with textual data Hamming Distance best suits as the similarity measure. Hamming distance between two strings is the number of positions at which the corresponding character are different. In other words, it measures the minimum number of substitutions
required to change one string to the other or the minimum number of errors that could have transformed one string into the other.
When a new sample is to be classified, we consider a majority vote of its neighbors, with the sample being assigned to the class that is most common amongst its K nearest neighbors measured using Hamming Distance.

<img src="https://raw.githubusercontent.com/vamshedhar/YelpReviewImages/master/KNN.gif" alt="K-Nearesrt Neghboure" width="400" />

### Logistic Regression
Logistic regression is a simple classification algorithm for learning to predict a discrete variable such as predicting whether a grid of pixel intensities represents a “0” digit or a “1” digit. Here we use a hypothesis class to try to predict the probability that a given sample belongs to the class “1” versus the probability that it belongs to the class “0”. Specifically, we will try to learn a function of the form:
<center>P(y=1|x) = h<sub>θ</sub>(x) = 1/1+exp(-θ<sup>T</sup>x)</center>
<center>P(y=0|x) = 1- P(y=1|x)</center>
The function 1/1+exp(-z) is often called the “sigmoid” or “logistic” function – it is an S-shaped function that “squashes” the value of z into the range [0,1] so that we may interpret h<sub>θ</sub>(x) as a probability. Our goal is to search for a value of θ so that the probability P(y=1|x)=h<sub>θ</sub>(x) is large when x belongs to the “1” class and small when x belongs to the “0” class (so that P(y=0|x) is large). We can learn to classify our training data by minimizing the cost function to find the best choice of θ. To estimate the θ we used following equation:	
		
<center>θ = θ + alpha * Σ x<sub>i</sub> (hθ(x<sub>i</sub>) − y<sub>i</sub>)</center>
To perform multi-class Logistic Regression, we used one Vs all strategy. This strategy involves training a single classifier per class, with the samples of that class as positive samples and all other samples as negatives. While predicting a class for test sample we apply all classifiers to the unseen sample and predict the label k for which the corresponding classifier reports the highest probability score.

<img src="https://raw.githubusercontent.com/vamshedhar/YelpReviewImages/master/LR.png" alt="Logistic Regression" width="400" />

## Results and Analysis
We built models for both 3-class and 5-class classification problem using above 3
supervised learning models. For 3-class classification, we programmatically constructed Good, Average and Bad classes from review labels.

### Naive Bayes Output

#### 3-Class output

<img src="https://raw.githubusercontent.com/vamshedhar/YelpReviewImages/master/NB%203.png" alt="Naive Bayes 3 Class Output" width="400" />

#### 5-Class output

<img src="https://raw.githubusercontent.com/vamshedhar/YelpReviewImages/master/NB%205.png" alt="Naive Bayes 5 Class Output" width="400" />

### K- Nearest Neighbours Output

#### 3-Class output

<img src="https://raw.githubusercontent.com/vamshedhar/YelpReviewImages/master/KNN%203.png" alt="K- Nearest Neighbours 3 Class Output" width="400" />

#### 5-Class output

<img src="https://raw.githubusercontent.com/vamshedhar/YelpReviewImages/master/KNN%205.png" alt="K- Nearest Neighbours 5 Class Output" width="400" />

### Logistic Regression Output

#### 3-Class output

<img src="https://raw.githubusercontent.com/vamshedhar/YelpReviewImages/master/LR%203.png" alt="Logistic Regression 3 Class Output" width="400" />

#### 5-Class output

<img src="https://raw.githubusercontent.com/vamshedhar/YelpReviewImages/master/LR%205.png" alt="Logistic Regression 5 Class Output" width="400" />


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

