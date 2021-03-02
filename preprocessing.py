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
        # clean_x = process_sentence(x)
        vocab.update(x)
    return list(vocab)

def create_embeddings(vocab):
    embeddings = {}
    vocab_embeddings = []
    for word in vocab:
        embed = embeddings.get(word)
        if embed:
            vocab_embeddings.append(embed)
        else:
            vocab_embeddings.append(embeddings['UNK'])
    vocab_embeddings.append(embeddings['UNK'])
    return vocab_embeddings

def create_sentence_representation(sentences, vocab, vocab_embeddings):
    sentence_rep = []
    for s in sentences:
        sr = []
        for word in s.split():
            if word in vocab:
                sr.append(vocab.index(word))
            else:
                sr.append(len(vocab_embeddings) - 1)
        sentence_rep.append(torch.tensor(sr))

def create_labels_index_representation(labels):
    label_indices = {}
    label_representation = []
    count = 0
    for label in label:
        if label not in label_indices:
            label_indices[token] = count
            count += 1
    for label in labels:
        label_representation.append(label_indices[label])
    return label_indices, label_representation


# performs necessary preprocessing and return relevant data
def preprocess_pipeline(file_path: str):
    labels = []
    sentences = []
    vocabulary = []
    vocabulary_embed = []
    sentence_representation = []
    label_index = {}
    label_representation = []

    # open file (containing sentences) and read line by line
    with open(file_path) as fp:
        # read first line
        line = fp.readline()

        # loop to read all lines
        while line:
            # get current label
            curr_label = line.split()[0]

            # add label
            labels.append(curr_label)

            # clean current sentence
            curr_sentence = split_tokenize(line)[1:]
            curr_sentence = remove_stop(curr_sentence)
            curr_sentence = remove_punc(curr_sentence)

            # add sentence
            sentences.append(curr_sentence)

            # read next line
            line = fp.readline()

    # generate vocabulary
    vocabulary = create_vocab(sentences)

    # generate the embeddings wrt vocabulary
    vocabulary_embed = create_embeddings(vocabulary)

    # generate sentence representations using vocab
    sentence_representation = create_sentence_representation(sentences, vocabulary, vocabulary_embed)

    # generate label indices and label representation
    label_index, label_representation = create_labels_index_representation(labels)

    return labels, sentences, vocabulary, vocabulary_embed, sentence_representation, label_index, label_representation

