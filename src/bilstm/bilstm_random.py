import torch.nn as nn
import torch
from bilstm.bilstm_model import BilstmModel

class BilstmRandom(BilstmModel):
    def __init__(self, input_size, hidden_zie, vocabulary_size, forward_hidden_zie, forward_output_size, enable_grad=True):
        super().__init__(input_size, hidden_zie, forward_hidden_zie, forward_output_size, enable_grad)
        self.vocabulary_size = vocabulary_size

        self.embedLayer = nn.Embedding(self.vocabulary_size, self.input_size)
        self.embedLayer.weight.requires_grad = self.enable_grad

class BilstmRandomEnsemble(nn.Module):
    def __init__(self, n_models, input_size, hidden_zie, vocabulary_size, forward_hidden_zie, forward_output_size, enable_grad=True):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            BilstmRandom(input_size, hidden_zie, vocabulary_size, forward_hidden_zie, forward_output_size, enable_grad) for _ in range(self.n_models)])

    def forward(self, x, l):
        results = []
        for model in self.models:
            results.append(model(x, l))
        return torch.stack(results).sum(dim=0)
