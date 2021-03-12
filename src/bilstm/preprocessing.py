import torch
import numpy as np
from string import punctuation
import re

NLTK_STOP_WORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
MORE_STOP_WORDS = set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])
ALL_STOP_WORDS = NLTK_STOP_WORDS.union(MORE_STOP_WORDS)

TO_REMOVE_STOP_WORDS = set(["where", "what", "who", "how", "when", "which", "why"])
ALL_STOP_WORDS = ALL_STOP_WORDS.difference(TO_REMOVE_STOP_WORDS)

# For spliting sentance into words , using general split
# converts to lower case as well
def split_tokenize(X: str):
    # converting to lower case and then splitting
    return X.lower().split()

#function to remove stop words
def remove_stop(word_list: list):
    # for removing stop words from dictionary list
    list_without_stop = [word for word in word_list if not word in ALL_STOP_WORDS]
    return list_without_stop

#function to remove punctuations
def remove_punc(word_list: list):
    # initializing punctuations string  
    list_without_punc = [re.sub(r'[{}]+'.format(punctuation), '', word) for word in word_list]
    return list_without_punc

# Wraper function that uses all functionality above on a sentence 
def process_sentence(sentence: str):
    #first tokenize 
    tokens = split_tokenize(sentence)
    #strips puncuation and stop words
    token_without_punc = remove_punc(tokens)
    clean_tokens = remove_stop(token_without_punc)
    return clean_tokens

# Function for creating Vocabulary
# Takes array of questions
# returns list of all words (no duplicates)
def create_vocab(data: list):
    #creating set in order to avoid duplicates
    vocab = set()
    for x in data:
        # clean_x = process_sentence(x)
        vocab.update(x.split())
    return list(vocab)

def create_embeddings(vocab: list, pre_train_file_path: str, unk_token: str):
    embeddings = {}
    with open(pre_train_file_path, 'r') as fp:
        line = fp.readline()
        while line:
            split_line = line.split()
            embeddings[split_line[0]] = np.array([float(x) for x in split_line[1:]])
            line = fp.readline()
    vocab_embeddings = []
    for word in vocab:
        embed = embeddings.get(word, np.array([]))
        if embed.size > 0:
            vocab_embeddings.append(embed)
        else:
            vocab_embeddings.append(embeddings[unk_token])
    vocab_embeddings.append(embeddings[unk_token])
    return vocab_embeddings

def create_sentence_representation(sentences: list, vocab: list, vocab_embeddings: list):
    sentence_rep = []
    for s in sentences:
        sr = []
        for word in s.split():
            if word in vocab:
                sr.append(vocab.index(word))
            else:
                sr.append(len(vocab_embeddings) - 1)
        sentence_rep.append(torch.tensor(sr))
    return sentence_rep

def create_labels_index(labels: list):
    label_indices = {}
    count = 0
    for label in labels:
        if label not in label_indices:
            label_indices[label] = count
            count += 1
    return label_indices

def create_labels_representation(labels: list, label_indices: dict):
    label_representation = []
    for label in labels:
        label_representation.append(label_indices[label])
    return label_representation


# performs necessary preprocessing and return relevant data
def preprocess_pipeline(file_path: str, pre_train_file_path: str, unk_token: str, is_train: bool):
    labels = []
    sentences = []
    vocabulary = []
    vocabulary_embed = []
    sentence_representation = []
    label_index = {}
    label_representation = []

    # open file (containing sentences) and read line by line
    with open(file_path, "r") as fp:
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
            sentences.append(' '.join(curr_sentence))

            # read next line
            line = fp.readline()

    if is_train:
        # generate vocabulary
        vocabulary = create_vocab(sentences)
        # generate the embeddings wrt vocabulary
        vocabulary_embed = create_embeddings(vocabulary, pre_train_file_path, unk_token)
        # generate label indices and label representation
        label_index = create_labels_index(labels)
    else:
        vocabulary = torch.load("../data/vocabulary.bin")
        vocabulary_embed = torch.load("../data/vocabulary_embed.bin")
        label_index = torch.load("../data/label_index.bin")

    # generate sentence representations using vocab
    sentence_representation = create_sentence_representation(sentences, vocabulary, vocabulary_embed)

    # generate label representation
    label_representation = create_labels_representation(labels, label_index)
    
    to_save = [
        (labels, 'labels'), 
        (sentences, 'sentences'), 
        (vocabulary, 'vocabulary'),
        (vocabulary_embed, 'vocabulary_embed'),
        (sentence_representation, 'sentence_representation'),
        (label_index, 'label_index'),
        (label_representation, 'label_representation')
    ]

    for bin_save in to_save:
        torch.save(bin_save[0], f"../data/{bin_save[1]}.bin")


def reload_preprocessed():
    to_load = {
        'labels': [], 
        'sentences': [], 
        'vocabulary': [],
        'vocabulary_embed': [],
        'sentence_representation': [],
        'label_index': {},
        'label_representation': []
    }

    for loaded in to_load.keys():
        to_load[loaded] = torch.load(f"../data/{loaded}.bin")

    return to_load['labels'], to_load['sentences'], to_load['vocabulary'], to_load['vocabulary_embed'], to_load['sentence_representation'], to_load['label_index'], to_load['label_representation']


class PreProcesseData:
    def __init__(self, file_path: str, pre_train_file_path: str, unk_token: str, is_train: bool):
        preprocess_pipeline(file_path, pre_train_file_path, unk_token, is_train=is_train)
        self.labels, self.sentences, self.vocabulary, self.vocabulary_embed, self.sentence_representation, self.label_index, self.label_representation = reload_preprocessed()