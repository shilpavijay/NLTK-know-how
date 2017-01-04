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

		conf = (votes.count(mode(votes))) / len(votes)
		return conf

short_pos = open("short_review/positive.txt","r").read()
short_neg = open("short_review/negative.txt","r").read()

document = []
all_words =[]

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for each in short_pos.split('\n'):
	document.append((each,'pos'))
	words = word_tokenize(each)
	pos = nltk.pos_tag(words)
	for w in pos:
		#allow only adjectives - meaningful to distinguish in the case of movie reviews
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

for each in short_neg.split('\n'):
	document.append((each,'neg'))
	words = word_tokenize(each)
	pos = nltk.pos_tag(words)
	for w in pos:
		#allow only adjectives - meaningful to distinguish in the case of movie reviews
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

save_docs = open("short_rev_docs.pickle","wb")
pickle.dump(document,save_docs)
save_docs.close()

#Get the frequency distribution of each word in the entire movie review
all_words = nltk.FreqDist(all_words)

#limit the words to first 5000 random reviews (+ve and -ve) - getting a list here
word_features = list(all_words.keys())[:5000]

save_wrd_feats = open("short_rev_wrdfeats.pickle","wb")
pickle.dump(word_features,save_wrd_feats)
save_wrd_feats.close()

#function to compare the above list of words with the dictionary(documents). If the word in word_features is present in the 
# dictionary(documents), the value is True else it is False.
def find_features(dictionary):
	dictwords = word_tokenize(dictionary)
	features = {}
	for w in word_features:
		features[w] = (w in dictwords)
	return features

feature_sets = [(find_features(rev),category) for (rev,category) in document] 

save_feat_sets = open("short_rev_feat_sets.pickle","wb")
pickle.dump(feature_sets,save_feat_sets)
save_feat_sets.close()

random.shuffle(feature_sets)

training_set=feature_sets[:10000]
testing_set=feature_sets[10000:]

# using Naive Bayes algorithm to train
classifier = nltk.NaiveBayesClassifier.train(training_set)

save_NBClass = open("short_rev_NBC.pickle","wb")
pickle.dump(classifier,save_NBClass)
save_NBClass.close()

print("Naive Bayes Algorithm accuracy percentage: ",nltk.classify.accuracy(classifier,testing_set)*100)
# # classifier.show_most_informative_features(25)


# # Multinomial Naive Bayes: (MNB)

# # MNB_classifier = SklearnClassifier(MultinomialNB())
# # MNB_classifier.train(training_set)
# # print("Multinomial NB Algorithm accuracy percentage: ",nltk.classify.accuracy(MNB_classifier,testing_set)*100)


#Instead of writing the same code for each NB Algorithm, here is a function:
# def SklClassifiers(ClassifierName):
# 	Classifier = SklearnClassifier(ClassifierName)
# 	Classifier.train(training_set)
# 	print(str(ClassifierName).partition('(')[0], "Algorithm Accuracy Percentage: ",nltk.classify.accuracy(Classifier,testing_set)*100)


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print( "Multinomial NB Accuracy Percentage: ",nltk.classify.accuracy(MNB_classifier,testing_set)*100)

save_MNBClass = open("short_rev_MNBC.pickle","wb")
pickle.dump(MNB_classifier,save_MNBClass)
save_MNBClass.close()

########################

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print( "Bernaulli NB Accuracy Percentage: ",nltk.classify.accuracy(BNB_classifier,testing_set)*100)

save_BNBClass = open("short_rev_BNBC.pickle","wb")
pickle.dump(BNB_classifier,save_BNBClass)
save_BNBClass.close()

##########################

LR_classifier = SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)
print( "Linear Regression Algorithm Accuracy Percentage: ",nltk.classify.accuracy(LR_classifier,testing_set)*100)

save_LRClass = open("short_rev_LR.pickle","wb")
pickle.dump(LR_classifier,save_LRClass)
save_LRClass.close()

###########################

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print( "SGDC Algorithm Accuracy Percentage: ",nltk.classify.accuracy(SGDC_classifier,testing_set)*100)

save_SGDCClass = open("short_rev_SGDC.pickle","wb")
pickle.dump(SGDC_classifier,save_SGDCClass)
save_SGDCClass.close()

###########################

LinerSVC_classifier = SklearnClassifier(LinearSVC())
LinerSVC_classifier.train(training_set)
print( "Linear SVC Algorithm Accuracy Percentage: ",nltk.classify.accuracy(LinerSVC_classifier,testing_set)*100)

save_LRSVCClass = open("short_rev_LRSVC.pickle","wb")
pickle.dump(LinerSVC_classifier,save_LRSVCClass)
save_LRSVCClass.close()

###########################

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print( "Numeric SVC Algorithm Accuracy Percentage: ",nltk.classify.accuracy(NuSVC_classifier,testing_set)*100)

save_NUSVClass = open("short_rev_NUSVC.pickle","wb")
pickle.dump(NuSVC_classifier,save_NUSVClass)
save_NUSVClass.close()

###########################

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