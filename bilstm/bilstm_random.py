import torch.nn as nn
from bilstm.bilstm_model import BilstmModel

class BilstmRandom(BilstmModel):
    def __init__(self, input_size, hidden_zie, vocabulary_size, forward_hidden_zie, forward_output_size, enable_grad=True):
        super().__init__(input_size, hidden_zie, forward_hidden_zie, forward_output_size, enable_grad)
        self.vocabulary_size = vocabulary_size

        self.embedLayer = nn.Embedding(self.vocabulary_size, self.input_size)
        self.embedLayer.weight.requires_grad = self.enable_grad