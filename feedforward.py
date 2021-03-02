import torch

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_zie, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_zie = hidden_zie
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_zie)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_zie, output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        # output = self.sigmoid(output)
        return output