import torch
from preprocessing import preprocess_pipeline, reload_preprocessed
from bilstm_random import BilstmRandom
from concat_dataset import ConcatDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
from eval import get_accuracy_bilstm, get_confusion_matrix, get_macro_f1, get_micro_f1, get_confusion_matrix

def qc_collate_fn_bilstm(input_dataset):
    length, data, label = [],[],[]
    for dataset in input_dataset:
        data.append(dataset[0])
        label.append(dataset[1])
        length.append(len(dataset[0]))
    data = torch.nn.utils.rnn.pad_sequence(data, padding_value=0)
    return data, label, length

load_trained = True

config = {
    "batch_size": 200,
    "embed_dim": 300,
    "bilstm_hidden_size": 100,
    "hidden_size": 300,
    "lr": 1,
    "momentum": 0,
    "epoch": 30,
    "early_stop": 500
}

if not load_trained:
    preprocess_pipeline("res/train_5500.label")

labels, sentences, vocabulary, vocabulary_embed, sentence_representation, label_index, label_representation = reload_preprocessed() 

# define train test split
train_qty = int(0.9 * len(labels))
validation_qty = len(labels) - train_qty

torch.manual_seed(0)

labelled_data = [[s, l] for s,l in zip(sentence_representation, label_representation)]
train_data, validation_data = torch.utils.data.random_split(labelled_data, [train_qty, validation_qty])

x_train, y_train = [], []

for train_dp in train_data:
    x_train.append(train_dp[0])
    y_train.append(train_dp[1])

x_validation, y_validation = [], []

for val_dp in validation_data:
    x_validation.append(val_dp[0])
    y_validation.append(val_dp[1])

# initialise dataloaders
concat_train = ConcatDataset((x_train, y_train))
dataloader_train = DataLoader(concat_train, batch_size=config["batch_size"], collate_fn=qc_collate_fn_bilstm)
concat_validation = ConcatDataset((x_validation, y_validation))
dataloader_validation = DataLoader(concat_validation, batch_size=config["batch_size"], collate_fn=qc_collate_fn_bilstm)

# initialise model
model = BilstmRandom(
    input_size=config['embed_dim'],
    hidden_zie=config['bilstm_hidden_size'],
    vocabulary_size=len(vocabulary) + 1,
    forward_hidden_zie=config['hidden_size'],
    forward_output_size=len(label_index)
)

loss_fxn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

model.train()
early_stop, best_accuracy = 0, 0

for epoch in range(config["epoch"]):
    batch_count = 1

    for data, label, length in dataloader_train:
        optimizer.zero_grad()
        y_pred = model(data, length)
        loss = loss_fxn(y_pred, torch.tensor(label))
        batch_count += 1
        loss.backward()
        optimizer.step()

        accuracy, _, _ = get_accuracy_bilstm(model, dataloader_validation)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            early_stop = 0
            torch.save(model, "data/bilstm.model")
            print(f"epoch: {epoch + 1}\tbatch: {batch_count}\taccuracy: {best_accuracy}")
        else:
            early_stop += 1
        if early_stop >= config["early_stop"]:
            print("early stop condition met")
            break

model = torch.load("data/bilstm.model")

accuracy, y_actual, y_pred = get_accuracy_bilstm(model, dataloader_validation)

confusion_matrix = get_confusion_matrix(y_actual, y_pred, len(label_index))

micro_f1 = get_micro_f1(confusion_matrix)
macro_f1, f1 = get_macro_f1(confusion_matrix)
print( "The accuray after training is : ", accuracy)
print("Confusion Matrix:\n", confusion_matrix)
print("Micro F1: ", micro_f1)
print("Macro F1: ", macro_f1)