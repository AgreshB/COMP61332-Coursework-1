import torch.nn as nn
from bilstm.bilstm_model import BilstmModel

class BilstmPretrain(BilstmModel):
    def __init__(self, embed, hidden_zie, forward_hidden_zie, forward_output_size, enable_grad=True):
        _, input_size_embed = embed.size()
        super().__init__(input_size_embed, hidden_zie, forward_hidden_zie, forward_output_size, enable_grad)
        
        self.embedLayer = nn.Embedding.from_pretrained(embeddings=embed, freeze=self.enable_grad)