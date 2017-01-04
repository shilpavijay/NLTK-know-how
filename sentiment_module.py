#File Name: sentiment_module.py

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
import pickle
from nltk.tokenize import word_tokenize

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
			votes.append(v)
		return mode(votes)

	def confidence(self,features):
		votes=[]
		for eachCfr in self._classifiers:
			v=eachCfr.classify(features)
			votes.append(v)

		conf = ((votes.count(mode(votes))) / len(votes))*100
		return conf

doc_f=open("short_rev_docs.pickle","rb")
document = pickle.load(doc_f)
doc_f.close()

wrdfeats_f=open("short_rev_wrdfeats.pickle","rb")
word_features = pickle.load(wrdfeats_f)
wrdfeats_f.close()


#function to compare the above list of words with the dictionary(documents). If the word in word_features is present in the 
# dictionary(documents), the value is True else it is False.
def find_features(dictionary):
	dictwords = word_tokenize(dictionary)
	features = {}
	for w in word_features:
		features[w] = (w in dictwords)
	return features

feat_set_f = open("short_rev_feat_sets.pickle","rb")
feature_sets = pickle.load(feat_set_f)
feat_set_f.close()

random.shuffle(feature_sets)

training_set=feature_sets[:10000]
testing_set=feature_sets[10000:]

open_file = open("short_rev_NBC.pickle","rb")
classifier = pickle.load(open_file)
open_file.close()

# print("Naive Bayes Algorithm accuracy percentage: ",nltk.classify.accuracy(classifier,testing_set)*100)
# # classifier.show_most_informative_features(25)

open_file = open("short_rev_MNBC.pickle","rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

# print( "Multinomial NB Accuracy Percentage: ",nltk.classify.accuracy(MNB_classifier,testing_set)*100)

########################

open_file = open("short_rev_BNBC.pickle","rb")
BNB_classifier = pickle.load(open_file)
open_file.close()

# print( "Bernaulli NB Accuracy Percentage: ",nltk.classify.accuracy(BNB_classifier,testing_set)*100)

##########################

open_file = open("short_rev_LR.pickle","rb")
LR_classifier = pickle.load(open_file)
open_file.close()

# print( "Linear Regression Algorithm Accuracy Percentage: ",nltk.classify.accuracy(LR_classifier,testing_set)*100)

###########################


open_file = open("short_rev_SGDC.pickle","rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

# print( "SGDC Algorithm Accuracy Percentage: ",nltk.classify.accuracy(SGDC_classifier,testing_set)*100)

###########################

open_file = open("short_rev_LRSVC.pickle","rb")
LinerSVC_classifier = pickle.load(open_file)
open_file.close()

# print( "Linear SVC Algorithm Accuracy Percentage: ",nltk.classify.accuracy(LinerSVC_classifier,testing_set)*100)

###########################

open_file = open("short_rev_NUSVC.pickle","rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()

# print( "Numeric SVC Algorithm Accuracy Percentage: ",nltk.classify.accuracy(NuSVC_classifier,testing_set)*100)

###########################

voted_cfr = VoteClassifier(classifier, 
							MNB_classifier, 
							BNB_classifier,
							LR_classifier,
							SGDC_classifier,
							LinerSVC_classifier,
							NuSVC_classifier
							)

def sentiments(text):
	feats = find_features(text)
	return voted_cfr.classify(feats), voted_cfr.confidence(feats)

# print("Voted Classifier Accuracy Percentage: " ,(nltk.classify.accuracy(voted_cfr,testing_set))*100)
# print("Classification: ", voted_cfr.classify(testing_set[0][0]), "Confidence: ", voted_cfr.confidence(testing_set[0][0]))