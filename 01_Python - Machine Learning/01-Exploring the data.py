# Student: XUEMEI LIN
# DATE: 2023-03-21

#IMPORTANTE !!! THIS DOCUMENT IS ONLY FOR GOOGLE COLAB (FILE .ipynb)

# Libraries needed for NLP
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# Libraries needed for Tensorflow proccessing
import tensorflow as tf
import numpy as np
import random
import json

# load the intents.json file from your local device
from google.colab import files
files.upload()

# import our chat-bot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)
    
# print intents.
# tag = key, patterns = conversation, responses = answers
intents

words = []
classes = []
documents = []
ignore = ['?']
# loop through each sentence in the intents'partterns
for intent in intents['intents']:
  for pattern in intent['patterns']:
    # tokenize each and every word in the sentence
    w = nltk.word_tokenize(pattern)
    # add word to the words list
    words.extend(w)
    # add word(s) to documents    first - list of words  second - tag of dangerous
    documents.append((w, intent['tag']))
    # add tags to our classes list
    if intent['tag'] not in classes:
      classes.append(intent['tag'])
      

# perform stemminng and lower each words as well as remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

# removee duplicate classes
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)