import torch
from preprocessing import preprocess_pipeline, reload_preprocessed

load_trained = True

if not load_trained:
    preprocess_pipeline("res/train_5500.label")

labels, sentences, vocabulary, vocabulary_embed, sentence_representation, label_index, label_representation = reload_preprocessed()

# define train test split
train_qty = int(0.9 * len(sentence_representation))
validation_qty = len(sentence_representation) - train_qty

torch.manual_seed(0)

labelled_data = [[s, l] for s,l in zip(sentence_representation, label_representation)]
train_data, validation_data = torch.utils.data.random_split(labelled_data, [train_qty, validation_qty])

x_train, y_train = [], []

for i in range(train_qty):
    x_train.append(train_data[i][0])
    y_train.append(train_data[i][1])

x_validation, y_validation = [], []

for i in range(validation_qty):
    x_validation.append(validation_data[i][0])
    y_validation.append(validation_data[i][1])