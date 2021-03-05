import torch

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_zie, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_zie = hidden_zie
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_zie)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(self.hidden_zie, output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        hidden = self.fc1(x)
        tanh = self.tanh(hidden)
        output = self.fc2(tanh)
        # output = self.sigmoid(output)
        return output