import torch
from torch import nn

class Controller(nn.Module):
    def __init__(self, input_dim, output_dim, controller_dim):
        super(Controller, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.controller_dim = controller_dim

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, controller_dim)

        # Define the linear layers for the read and write heads
        self.read_head_layer = nn.Linear(controller_dim, output_dim)
        self.write_head_layer = nn.Linear(controller_dim, output_dim)

    def forward(self, input, prev_state):
        # Pass the input through the LSTM layer
        out, state = self.lstm(input.unsqueeze(0), prev_state)

        # Pass the LSTM output through the read and write head layers
        read_head = self.read_head_layer(out.squeeze(0))
        write_head = self.write_head_layer(out.squeeze(0))

        return read_head, write_head, state
