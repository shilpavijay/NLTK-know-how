from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

text = "Hello Mr.Watson, how are you doing? How is the weather today?"

# print sent_tokenize(text)
# ['Hello Mr.Watson, how are you doing?', 'How is the weather today?']


# print word_tokenize(text)
# ['Hello', 'Mr.Watson', ',', 'how', 'are', 'you', 'doing', '?', 'How', 'is', 'the', 'weather', 'today', '?']

# for i in word_tokenize(text):
# 	print i

exStopWords = "This is a sample example to exhibit stopwords"

stop_words=set(stopwords.words("english"))	
#complete set of stop words

words = word_tokenize(exStopWords)

expl_StopWords = [w for w in words if w not in stop_words]

print expl_StopWords
