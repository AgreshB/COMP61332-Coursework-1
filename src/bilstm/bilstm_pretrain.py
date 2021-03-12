import torch
import torch.nn as nn
from bilstm.bilstm_model import BilstmModel

class BilstmPretrain(BilstmModel):
    def __init__(self, embed, hidden_zie, forward_hidden_zie, forward_output_size, enable_grad=True):
        _, input_size_embed = embed.size()
        super().__init__(input_size_embed, hidden_zie, forward_hidden_zie, forward_output_size, enable_grad)
        
        self.embedLayer = nn.Embedding.from_pretrained(embeddings=embed, freeze=self.enable_grad)

class BilstmPretrainEnsemble(nn.Module):
    def __init__(self, n_models, embed, hidden_zie, forward_hidden_zie, forward_output_size, enable_grad=True):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            BilstmPretrain(embed, hidden_zie, forward_hidden_zie, forward_output_size, enable_grad) for _ in range(self.n_models)])

    def forward(self, x, l):
        results = []
        for model in self.models:
            results.append(model(x, l))
        return torch.stack(results).sum(dim=0)