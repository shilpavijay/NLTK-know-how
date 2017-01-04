NLTK - Natural Language ToolKit		
-------------------------------

NLTK helps build Python programs to work with human language. It is the Python library for NLP or Natural Language Processing.

Installation:
-------------

pip install nlk

pip install numpy

To test: Run python and import nltk

then, run nltk.download(). This may take quite some time to complete.


Other Libraries required:
--------------------------
1. Numpy
2. Scipy
3. Matplotlib
4. Scikit-learn

What to expect from this Repository ?
-----------------------------------

This Repository contains tiny Python Programs on many of the NLTK libraries. While learning NLTK, one can practise Tokenizers, Stemming, Stop Words, Chunking, Lemmatizers and many others.

This Repo also contains a project on SENTIMENT ANALYSIS based on Machine Learning Algorithms. 

The sentiment analysis is perfomed on Movie Reviews taken from the Corpora - "movie_reviews" which can be found in the folder '\AppData\Roaming\nltk_data\corpora' once NLTK is installed. 

7 different Machine Learning Algorithms are used for training. The end result is, given a new movie review, the module will output the Sentiment (classified as Positive/Negative) as well as the Confidence (which is based on the 7 results from 7 Machinee Learning Algorithms)

sentiment_module.py can be imported and used for Sentiment Analysis on any new Data Set.