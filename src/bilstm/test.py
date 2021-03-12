import torch
import numpy as np

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

    def accuracy_fxn(self, x):
        with torch.no_grad():
            y_pred = self.model(x, self.l).argmax(dim=1)
            return np.sum(y_pred.numpy() == self.y_actual) / len(self.y_actual), y_pred

    def doTesting(self):
        x = torch.nn.utils.rnn.pad_sequence(self.sentence_representation, padding_value=0)
        accuracy, y_pred = self.accuracy_fxn(x)
        return accuracy, y_pred
