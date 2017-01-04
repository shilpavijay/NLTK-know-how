import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode

class VoteClassifier(ClassifierI):
	def __init__(self,*classifiers):
		self._classifiers = classifiers

	def classify(self,features):
		votes=[]
		for eachCfr in self._classifiers:
			v=eachCfr.classify(features)
			# print(v)
			votes.append(v)
		return mode(votes)

	def confidence(self,features):
		votes=[]
		for eachCfr in self._classifiers:
			v=eachCfr.classify(features)
			# print(v)
			votes.append(v)

		conf = votes.count(mode(votes)) / len(votes)
		return conf




get_docs = open("MovieRevDocs.pickle","rb")
documents = pickle.load(get_docs)
get_docs.close()
# documents = [ (list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

#shuffling is required as the movie review data is sorted - first all positive and then all negetive.
random.shuffle(documents)

all_words = []

#obtaining only the words in the movie reviews - in lower case (to have a single copy of any given word)
for words in movie_reviews.words():
	all_words.append(words.lower())

#Get the frequency distribution of each word in the entire movie review
all_words = nltk.FreqDist(all_words)

#limit the words to first 3000 random reviews (+ve and -ve) - getting a list here
word_features = list(all_words.keys())[:3000]

#function to compare the above list of words with the dictionary(documents). If the word in word_features is present in the 
# dictionary(documents), the value is True else it is False.
def find_features(dictionary):
	dictwords = set(dictionary)
	features = {}
	for w in word_features:
		features[w] = (w in dictwords)
	return features

feature_sets = [(find_features(rev),category) for (rev,category) in documents] 

training_set=feature_sets[:1900]
testing_set=feature_sets[1900:]

open_cfr = open("BayesClassifier.pickle","rb")
classifier = pickle.load(open_cfr)
open_cfr.close()
#using Naive Bayes algorithm to train
# classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Naive Bayes Algorithm accuracy percentage: ",nltk.classify.accuracy(classifier,testing_set)*100)
# classifier.show_most_informative_features(25)


# Multinomial Naive Bayes: (MNB)

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print("Multinomial NB Algorithm accuracy percentage: ",nltk.classify.accuracy(MNB_classifier,testing_set)*100)


#Instead of writing the same code for each NB Algorithm, here is a function:
def SklClassifiers(ClassifierName):
	Classifier = SklearnClassifier(ClassifierName)
	Classifier.train(training_set)
	print(str(ClassifierName).partition('(')[0], "Algorithm Accuracy Percentage: ",nltk.classify.accuracy(Classifier,testing_set)*100)

# Multinomial Naive Bayes: (MNB)
SklClassifiers(MultinomialNB())

#Bernoulli Naive Bayes: (BNB)
SklClassifiers(BernoulliNB())

# LogisticRegression:
SklClassifiers(LogisticRegression())

SklClassifiers(SGDClassifier())
SklClassifiers(SVC())
SklClassifiers(LinearSVC())
SklClassifiers(NuSVC())


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)

LR_classifier = SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)

LinerSVC_classifier = SklearnClassifier(LinearSVC())
LinerSVC_classifier.train(training_set)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

voted_cfr = VoteClassifier(classifier, 
							MNB_classifier, 
							BNB_classifier,
							LR_classifier,
							SGDC_classifier,
							LinerSVC_classifier,
							NuSVC_classifier
							)

print("Voted Classifier Accuracy Percentage: " ,(nltk.classify.accuracy(voted_cfr,testing_set))*100)
print("Classification: ", voted_cfr.classify(testing_set[0][0]), "Confidence: ", voted_cfr.confidence(testing_set[0][0]))