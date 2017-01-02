from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

sentence = "It is important to read properly. Many people make mistakes while reading. Manu reads well but should still improve in reading."

words = word_tokenize(sentence)

for each in words:
	print (ps.stem(each))