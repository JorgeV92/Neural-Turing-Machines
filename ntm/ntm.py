import torch
import torch.nn as nn
from controller import Controller
from memory import Memory

class NTM(nn.Module):
    def __init__(self, controller_type, input_size, output_size, hidden_size, N, M):
        super(NTM, self).__init__()
        self.controller = Controller(controller_type, input_size + M, output_size, hidden_size)
        self.memory = Memory(N, M)
        self.read_weights = torch.zeros(N)
        self.write_weights = torch.zeros(N)

    def forward(self, x):
        # Read from memory
        read_vector = self.memory.read(self.read_weights)

        # Pass input and read vector to controller
        out, state = self.controller(torch.cat([x, read_vector]), self.controller.init_state(x.size(0)))

        # Calculate new read and write weights
        # This will depend on the specifics of your project
        # For example, you might use a softmax function to ensure the weights sum to 1
        self.read_weights = torch.softmax(out[:, :self.memory.N], dim=1)
        self.write_weights = torch.softmax(out[:, self.memory.N:2*self.memory.N], dim=1)

        # Write to memory
        erase_vector = torch.sigmoid(out[:, 2*self.memory.N:3*self.memory.N])
        add_vector = torch.tanh(out[:, 3*self.memory.N:])
        self.memory.write(self.write_weights, erase_vector, add_vector)

        return out
