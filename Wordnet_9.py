from nltk.corpus import wordnet

syns = wordnet.synsets("program")

print(syns[0])
#Synset('plan.n.01')

print(syns[0].name())
#plan.n.01

print(syns[0].lemmas()[0].name())
#plan

print(syns[0].definition())
#a series of steps to be carried out or goals to be accomplished

print(syns[0].examples())
#['they drew up a six-step plan', 'they discussed plans for a new bond issue']

#FINDING SYNONYMS AND ANTONYMS FOR 'good':

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
	for l in syn.lemmas():
		synonyms.append(l.name())
		if l.antonyms():
			antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

print("")
print("Finding similarities between two words")

word1=wordnet.synset("ship.n.01")
word2=wordnet.synset("boat.n.01")

print(word1.wup_similarity(word2))


word1=wordnet.synset("ship.n.01")
word2=wordnet.synset("car.n.01")

print(word1.wup_similarity(word2))

word1=wordnet.synset("ship.n.01")
word2=wordnet.synset("green.n.01")

print(word1.wup_similarity(word2))