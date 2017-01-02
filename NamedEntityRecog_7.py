
#Named Entities - Organization, Person, Date, Time, Money, Percentage, Facility, Location, GPE
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

training_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(training_text)

token = custom_sent_tokenizer.tokenize(sample_text)

def process_givenData():
	try:
		for i in token:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)    #POS - PARTS OF SPEECH

			NamedEnt = nltk.ne_chunk(tagged, binary=True)
			NamedEnt.draw()	
	except Exception as e:
		print('i am in exception')


process_givenData()