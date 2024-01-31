import torch
import torch.nn as nn


# Create a simple MLP Neural Network to approximate f
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_hidden_layers):
        super().__init__()
        # n_hidden_layers: number of hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output(x)
        return x
