from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

exStopWords = "This is a sample example to exhibit stopwords"

stop_words=set(stopwords.words("english"))	
#complete set of stop words

words = word_tokenize(exStopWords)

expl_StopWords = [w for w in words if w not in stop_words]

print(expl_StopWords)
