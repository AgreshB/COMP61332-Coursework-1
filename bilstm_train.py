import torch
from preprocessing import preprocess_pipeline, reload_preprocessed

load_trained = True

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