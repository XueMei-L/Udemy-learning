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