import torch
from torch import nn


class MLP(nn.Module): 

    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(MLP, self).__init__()

        # FC
        self.input_fc = nn.Linear(input_size * input_size, hidden_size) 
        self.output_fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()


    def forward(self, x):

        batch_size = x.shape[0]
        x1 = self.input_fc(x.view(batch_size, -1))
        x1 = self.relu(x1)

        x2 = self.output_fc(x1)

        # DO NOT ACTIVATE x2 because we are using bcewithlogits

        return x2

