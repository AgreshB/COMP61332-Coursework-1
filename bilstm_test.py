import torch
from preprocessing import preprocess_pipeline, reload_preprocessed
from eval import get_accuracy_test, get_accuracy_bilstm, get_confusion_matrix, get_macro_f1, get_micro_f1, get_confusion_matrix

load_trained = False

config = {
    "batch_size": 200,
    "embed_dim": 300,
    "bilstm_hidden_size": 100,
    "hidden_size": 300,
    "lr": 1,
    "momentum": 0,
    "epoch": 30,
    "early_stop": 500,
    "path_model": "data/bilstm.model"
}

if not load_trained:
    preprocess_pipeline("res/TREC_10.label", is_train=False)

labels, sentences, vocabulary, vocabulary_embed, sentence_representation, label_index, label_representation = reload_preprocessed() 

y_actual = label_representation
lengths = []
for s in sentence_representation:
    lengths.append(len(s))

model = torch.load(config["path_model"])

x = torch.nn.utils.rnn.pad_sequence(sentence_representation ,padding_value=0)
accuracy, y_pred = get_accuracy_test(model, "bilstm", x, y_actual, lengths)
confusion_matrix = get_confusion_matrix(y_actual, y_pred, len(label_index))

micro_f1 = get_micro_f1(confusion_matrix)
macro_f1, f1 = get_macro_f1(confusion_matrix)
print( "Accuracy: ", accuracy)
print("Confusion Matrix:\n", confusion_matrix)
print("Micro F1: ", micro_f1)
print("Macro F1: ", macro_f1)