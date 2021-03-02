import torch
from feedforward import Feedforward

class BilstmRandom(torch.nn.Module):
    def __init__(self, input_size, hidden_zie, vocabulary_size, forward_hidden_zie, forward_output_size, enable_grad=False):
        super(BilstmRandom, self).__init__()
        self.input_size = input_size
        self.hidden_zie = hidden_zie
        self.forward_hidden_zie = forward_hidden_zie
        self.forward_output_size = forward_output_size
        self.enable_grad = enable_grad
        self.vocabulary_size = vocabulary_size

        self.embedLayer = torch.nn.Embedding(self.vocabulary_size, self.input_size)
        self.embedLayer.weight.requires_grad = self.enable_grad

        self.bilstm = torch.nn.LSTM(self.input_size, self.hidden_zie, bidirectional=True)
        self.feedforwardnet = Feedforward((2 * self.hidden_zie), self.forward_hidden_zie, self.forward_output_size)

        self.log_softmax = torch.nn.LogSoftmax(dim=1)