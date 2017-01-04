import nltk
import random
from nltk.corpus import movie_reviews
import pickle

# documents = [ (list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
#'documents would act like a dictionary here'

#commented out the above documents after generating it once and pickling it to MovieRevDocs.pickle.
#Now fetching the saved documents using pickle
get_docs = open("MovieRevDocs.pickle","rb")
documents = pickle.load(get_docs)
get_docs.close()

#save the documents using pickle
# save_doc = open("MovieRevDocs.pickle","wb")
# pickle.dump(documents,save_doc)
# save_doc.close()

#shuffling is required as the movie review data is sorted - first all positive and then all negetive.
random.shuffle(documents)

# print(documents[1])

all_words = []

#obtaining only the words in the movie reviews - in lower case (to have a single copy of any given word)
for words in movie_reviews.words():
	all_words.append(words.lower())

# print(all_words[2:10])

#Get the frequency distribution of each word in the entire movie review
all_words = nltk.FreqDist(all_words)

# print(all_words.most_common(25))
"""
[(',', 77717), ('the', 76529), ('.', 65876), ('a', 38106), ('and', 35576), ('of', 34123), ('to', 31937), ("'", 30585), 
('is', 25195), ('in', 21822), ('s', 18513), ('"', 17612), ('it', 16107), ('that', 15924), ('-', 15595), (')', 11781), 
('(', 11664), ('as', 11378), ('with', 10792), ('for', 9961), ('his', 9587), ('this', 9578), ('film', 9517), ('i', 8889), 
('he', 8864)]
"""
# print(all_words["bad"])
#1395
# print(all_words["excellent"])
#184

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

# sample_features=find_features(movie_reviews.words('neg/cv000_29416.txt'))
# {each for each in sample_features if sample_features[each] == "False"}

feature_sets = [(find_features(rev),category) for (rev,category) in documents] 

training_set=feature_sets[:1900]
testing_set=feature_sets[1900:]

#using Naive Bayes algorithm to train
# classifier = nltk.NaiveBayesClassifier.train(training_set)

#after saving the classifier using pickle, commented out the above. Now fetching the same using pickle
open_cfr = open("BayesClassifier.pickle","rb")
classifier = pickle.load(open_cfr)
open_cfr.close()

print("Naive Bayes Algorithm accuracy percentage: ",nltk.classify.accuracy(classifier,testing_set)*100)
classifier.show_most_informative_features(25)

#save the classifier using pickle
# save_classifier = open("BayesClassifier.pickle","wb")
# pickle.dump(classifier,save_classifier)
# save_classifier.close()