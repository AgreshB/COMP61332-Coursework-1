import torch
from bilstm.preprocessing import PreProcesseData
import numpy as np

def get_accuracy_test(model, model_type, x, y, lengths):
    with torch.no_grad():
        if model_type=='bow':
            y_preds = model(x).argmax(dim=1)
            return np.sum(y_preds.numpy()==y)/len(y),y_preds
        if model_type=='bilstm':
            y_preds = model(x,lengths).argmax(dim=1)
            return np.sum(y_preds.numpy()==y)/len(y),y_preds
        if model_type=='bow_bilstm':
            y_preds = model(x,lengths).argmax(dim=1)
            return np.sum(y_preds.numpy()==y)/len(y),y_preds

class Test:
    def __init__(self, preProcessedData, model, model_type: str):
        super().__init__()
        self.model = model
        self.y_actual = preProcessedData.label_representation
        self.sentence_representation = preProcessedData.sentence_representation
        self.l = []
        self.model_type = model_type
        for s in self.sentence_representation:
            self.l.append(len(s))

    def doTesting(self):
        x = torch.nn.utils.rnn.pad_sequence(self.sentence_representation, padding_value=0)
        accuracy, y_pred = get_accuracy_test(self.model, self.model_type, x, self.y_actual, self.l)
        return accuracy, y_pred
