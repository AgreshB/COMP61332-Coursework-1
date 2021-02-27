import torch
from torch import nn
from collections import defaultdict
import time
import matplotlib.pyplot as plt

# creating list of stop words from Stop_words.txt
STOP_WORDS = []
with open("res/stop_words.txt") as f:
    for line in f:
        STOP_WORDS.append(line.strip())

# For spliting sentance into words , using general split
# converts to lower case as well
def split_tokenize(X):
    # converting to lower case and then splitting
    return X.lower().split()

#function to remove stop words
def remove_stop(word_list):
    # for removing stop words from dictionary list
    list_without_stop = [word for word in word_list if not word in STOP_WORDS]
    return list_without_stop

#function to remove punctuations
def remove_punc(word_list):
    # initializing punctuations string  
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    punc_list = list(punc)
    list_without_punc = [word for word in word_list if not word in punc_list]
    return list_without_punc


# Wraper function that uses all functionality above on a sentence 
def process_sentence(sentence):
    #first tokenize 
    tokens = split_tokenize(sentence)
    #strips puncuation and stop words
    token_without_punc = remove_punc(tokens)
    clean_tokens = remove_stop(token_without_punc)
    return clean_tokens

# Function for creating Vocabulary
# Takes array of questions
# returns list of all words (no duplicates)
def create_vocab(data):
    #creating set in order to avoid duplicates
    vocab = set()
    for x in data:
        clean_x = process_sentence(x)
        vocab.update(clean_x)
    return list(vocab)

