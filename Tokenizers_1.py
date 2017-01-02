from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

text = "Hello Mr.Watson, how are you doing? How is the weather today?"

print(sent_tokenize(text))
# ['Hello Mr.Watson, how are you doing?', 'How is the weather today?']


print(word_tokenize(text))
# ['Hello', 'Mr.Watson', ',', 'how', 'are', 'you', 'doing', '?', 'How', 'is', 'the', 'weather', 'today', '?']

for i in word_tokenize(text):
	print(i)

