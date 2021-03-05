
from classifier import Classifier
from torch import nn
import torch 
import numpy as np
from collections import defaultdict
import time
from preprocessing import process_sentence
import matplotlib.pyplot as plt


class BoWClassifierModule(nn.Module):
    def __init__(self, text_field_vocab, class_field_vocab, emb_dim, dropout=0.5):
        super().__init__()
        self.linear = nn.Linear(text_field_vocab, class_field_vocab)
    def forward(self, docs):
        return nn.functional.log_softmax(self.linear(docs), dim=0)

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel

class BoWTextClassifierModule(nn.Module):
    def __init__(self, text_field_vocab, class_field_vocab, emb_dim, dropout=0.5, pretrained=False):
        super().__init__()
        self.embedding = nn.Embedding(len(text_field_vocab), 300)
        if(pretrained):
            glove_model = loadGloveModel('data\glove.small')

            weights_matrix = np.zeros((len(text_field_vocab), 300))
            for idx, word in enumerate(text_field_vocab):
                try:
                    weights_matrix[idx] = glove_model[text_field_vocab[word]]
                except KeyError:
                    weights_matrix[idx] = np.random.normal(scale=0.6, size=(300, ))
            weights_matrix = torch.from_numpy(weights_matrix).to('cuda' if torch.cuda.is_available() else 'cpu')
            self.embedding.load_state_dict({'weight': weights_matrix})
        
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(300, len(class_field_vocab))
    
    def forward(self, docs):
        embedded = self.embedding(docs)
        bow = embedded.mean(dim=1)
        #bow_drop = self.dropout(bow)
        return nn.functional.log_softmax(self.lin(bow), dim=0)

# Read file data
def read_data(filename):
    with open(filename) as f:
        labels = []
        split_sentence = []
        word_vocabulary = []
        label_vocabulary = []
        for line in f:
            columns = line.strip().split(maxsplit=1)
            doc = (columns[-1])
            doc = (process_sentence(doc))
            label = columns[0]
            words = []
            for token in doc:
                if token not in word_vocabulary:
                    word_vocabulary.append(token)
                words.append(word_vocabulary.index(token))
            split_sentence.append(words)
            if label not in label_vocabulary:
                label_vocabulary.append(label)
            labels.append(label_vocabulary.index(label))
    return labels, split_sentence, label_vocabulary, word_vocabulary

def evaluate_validation(scores, loss_function, correct):
    guesses = scores.argmax(dim=1)
    n_correct = (guesses == correct).sum().item()
    return n_correct, loss_function(scores, correct).item()

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class BagOfWords(Classifier):
    def __init__(self, config):
        Classifier.__init__(self,config)
        self.config = config

        try:
            self.lr = float(config.get('BOW', 'lr_param'))
        except ValueError:
            print("lr_param not a float. Defaulting to 0.02...")
            self.lr = 0.02
        try:
            self.epoch = int(config.get('BOW', 'epoch'))
        except ValueError:
            print("epoch not an integer. Defaulting to 100...")
            self.epoch = 100
        try:
            self.emb = int(config.get('BOW', 'emb'))
        except ValueError:
            print("emb not a an integer. Defaulting to 16...")
            self.emb = 16
        try:
            self.batch_size = int(config.get('BOW', 'batch_size'))
        except ValueError:
            print("batch_size not a an integer. Defaulting to 128...")
            self.batch_size = 128
        try:
            self.stop_loss = int(config.get('BOW', 'stop_loss'))
        except ValueError:
            print("stop_loss not a an integer. Defaulting to 5...")
            self.stop_loss = 5
        try:
            self.min_word_freq = int(config.get('BOW', 'min_word_freq'))
        except ValueError:
            print("min_word_freq not a an integer. Defaulting to 5...")
            self.min_word_freq = 5

    def evaluate_validation(scores, loss_function, correct):
        guesses = scores.argmax(dim=1)
        n_correct = (guesses == correct).sum().item()
        return n_correct, loss_function(scores, correct).item()

    def train(self):
        print("BoW: Training began...")
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running device on:',device)
        label, data, label_vocab, data_vocab = read_data(self.train_filename)


        # Create tensors
        data_array = np.zeros([len(data_vocab),len(data)], dtype = np.double)
        for i,j in enumerate(data):
            for idx in j:
                data_array[idx][i] += 1

        # Remove words with a low frequency
        data_freq = data_array.sum(axis=1)
        zipped_lists = list(zip(data_freq, data_array))
        zipped_lists = sorted(zipped_lists, key = lambda x: x[0], reverse=True)
        zipped_lists = [(x, y) for x, y in zipped_lists if x >= self.min_word_freq]
        data_freq, data_array = zip(*zipped_lists)
        data_array = np.array(data_array).transpose()

        # Data to pytorch 
        data_array = torch.from_numpy(np.array(data_array, dtype=np.double)).to(device)
        label_array = torch.from_numpy(np.array(label, dtype=np.int64)).to(device)
        
        # Split training and validation 90/10
        validation_split = .1
        split = int(np.floor(validation_split * len(label_array)))
        train_label, val_label = label_array[split:], label_array[:split]
        train_data, val_data = data_array[split:], data_array[:split]
        
        # Declare the model
        #model = BoWTextClassifierModule((test), (label_vocab), emb_dim=self.emb)   
        model = BoWClassifierModule(len(data_freq), len(label_vocab), emb_dim=self.emb)  
        
        # Put the model on gpu/cpu
        model.to(device)

        # Cast floating points to double types
        model.double()
        
        # Cross-entropy loss and adam optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # History to record
        history = defaultdict(list)
        
        # Randomize training data
        train_loader = torch.utils.data.DataLoader(
            ConcatDataset(
                train_label,
                train_data
            ),
            batch_size=self.batch_size, shuffle=True)
        n_batches = len(train_loader)
        loss_count = 0
        prev_val_acc = 0.
        n_valid = len(val_data)
        # Iterate through epochs
        for i in range(self.epoch):
            # Reset variables
            t0 = time.time()
            loss_sum = 0

            # Enable the dropout layers.
            model.train()
            
            # Iterate through the batches created
            for (train_label_batch, train_data_batch) in (train_loader):

                # Compute the scores and loss function
                scores = model((train_data_batch))
                #scores = model(torch.transpose(train_data_batch))
                loss = loss_function(scores, train_label_batch)

                # Compute the gradient with respect to the loss, and update the parameters of the model.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Add loss
                loss_sum += loss.item()

            # Average out the loss
            train_loss = loss_sum / n_batches
            history['train_loss'].append(train_loss)

            # Disable the dropout layers.
            model.eval()
            
            # Compute the loss and accuracy on the validation set.
            scores = model(val_data)
            #scores = model(torch.transpose(val_data))
            correct, val_loss = evaluate_validation(scores, loss_function, val_label)
            val_acc = correct / n_valid
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # epoch timer end
            t1 = time.time()

            # Print epoch data
            print(f'Epoch {i+1}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, val acc: {val_acc:.4f}, time = {t1-t0:.4f}')
            if(val_acc < prev_val_acc):
                loss_count += 1
                if(loss_count >= self.stop_loss):
                    break
            else:
                loss_count = 0
                prev_val_acc = val_acc
        print("BoW: Training complete!")
        # Plot model
        plt.plot(history['train_loss'])
        plt.plot(history['val_loss'])
        plt.plot(history['val_acc'])
        plt.legend(['training loss', 'validation loss', 'validation accuracy'])
        plt.show()

    #TODO: Create Test function
    def test(self):
        print("BoW: Test results:")