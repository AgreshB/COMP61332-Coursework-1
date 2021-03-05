
from classifier import Classifier
from torch import nn
import torch 
import numpy as np
from collections import defaultdict
import time
from preprocessing import process_sentence,create_vocab,split_tokenize


class BoWClassifier(nn.Module):
    def __init__(self, vocab_size, class_field_vocab, emb_dim, dropout=0.5, pre_train_weights = None,freeze= True):
        super().__init__()
        if(pre_train_weights is not None):
            self.embedding = nn.Embedding.from_pretrained(pre_train_weights,freeze)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx = 0)
        self.top_layer = nn.Linear(emb_dim, class_field_vocab)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, docs):
        embedded = self.embedding(docs)
        cbow = embedded.mean(dim=0)
        cbow_drop = self.dropout(cbow)
        scores = self.top_layer(cbow_drop)
        return scores

# Read file data
def read_data(filename):
    X = []
    y = []
    with open(filename,encoding = "ISO-8859-1") as f:
        for line in f:
            label ,sentance = line.strip().split(maxsplit=1)
            X.append(sentance)
            y.append(label)
        split = int(0.9 * len(X))

    X_train = X[:split]
    X_val = X[split:]

    y_train = y[:split]
    y_val = y[split:]

    # creating vocabulary with sentances
    vocab = create_vocab(X)
    return X, y , vocab

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
        self.vocab = []
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

    def evaluate_validation(scores, loss_function, correct):
        guesses = scores.argmax(dim=1)
        n_correct = (guesses == correct).sum().item()
        return n_correct, loss_function(scores, correct).item()


    def train(self):
        print("BoW: Training began...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        data , label , self.vocab = read_data(self.train_filename)
    
        # Create tensors
        data_array = np.zeros([len(data),len(max(data,key = lambda x: len(x)))], dtype = np.int64)
        for i,j in enumerate(data):
            data_array[i][0:len(j)] = j
        data_array = torch.from_numpy(data_array).to(device)
        label_array = torch.from_numpy(np.array(label, dtype = np.int64)).to(device)
        
        # Split training and validation 90/10
        validation_split = .1
        split = int(np.floor(validation_split * len(label_array)))
        train_label, val_label = label_array[split:], label_array[:split]
        train_data, val_data = data_array[split:], data_array[:split]
        
        n_valid = len(val_data)
        n_clases =np.unique(np.array(label))

        # Declare the model
        model = BoWClassifier(len(vocab), len(n_clases), emb_dim=self.emb)   
        
        # Put the model on cpu
        # TODO: GPU
        device = 'cpu'
        model.to(device)
        
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
                scores = model((torch.transpose(train_data_batch, 0, 1)))
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
            scores = model(torch.transpose(val_data, 0, 1))
            correct, val_loss = evaluate_validation(scores, loss_function, val_label)
            val_acc = correct / n_valid
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # epoch timer end
            t1 = time.time()

            # Print epoch data
            print(f'Epoch {i+1}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, val acc: {val_acc:.4f}, time = {t1-t0:.4f}')
        print("BoW: Training complete!")

    #TODO: Create Test function
    def test(self):
        print("BoW: Test results:")
