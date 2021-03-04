import torch
import torch.nn as nn
from feedforward import Feedforward

class BilstmModel(nn.Module):
    def __init__(self, input_size, hidden_zie, forward_hidden_zie, forward_output_size, enable_grad=True):
        super(BilstmModel, self).__init__()
        self.input_size = input_size
        self.hidden_zie = hidden_zie
        self.forward_hidden_zie = forward_hidden_zie
        self.forward_output_size = forward_output_size
        self.enable_grad = enable_grad
        
        self.embedLayer = None

        self.bilstm = nn.LSTM(self.input_size, self.hidden_zie, bidirectional=True)
        self.feedforwardnet = Feedforward((2 * self.hidden_zie), self.forward_hidden_zie, self.forward_output_size)

        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, l):
        embed = self.embedLayer(x)
        pack_padded = torch.nn.utils.rnn.pack_padded_sequence(embed, l, enforce_sorted=False)

        pack_output, _ = self.bilstm(pack_padded)
        bilstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_output)
        bilstm_output_size = bilstm_output.size()[1]

        # calculate mean of (re)padded vectors
        for i in range(0, bilstm_output_size):
            if i == 0:
                vector = bilstm_output.index_select(
                        1,
                        torch.tensor(i)
                    ).squeeze(
                        1
                    ).index_select(
                        0,
                        torch.tensor(l[i]-1)
                    ).mean(
                        dim=0
                    ).unsqueeze(0)
            else:
                vector = torch.cat(
                    (
                        vector,
                        bilstm_output.index_select(
                            1,
                            torch.tensor(i)
                        ).squeeze(
                            1
                        ).index_select(
                            0,
                            torch.tensor(l[i]-1)
                        ).mean(
                            dim=0
                        ).unsqueeze(0)),
                    0)

        forwardfeednn = self.feedforwardnet(vector)

        return self.log_softmax(forwardfeednn)