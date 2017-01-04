from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("Booking"))
print(lemmatizer.lemmatize("formuli"))
print(lemmatizer.lemmatize("Spoke"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("inspecting"))

print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("running", "v"))