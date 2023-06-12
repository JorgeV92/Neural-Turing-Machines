import torch
import torch.nn as nn

class Memory(nn.Module):
    def __init__(self, N, M):
        super(Memory, self).__init__()
        self.N = N  # Number of memory locations
        self.M = M  # Size of each memory location
        self.memory = torch.zeros(N, M)  # Initialize memory matrix
        self.link_matrix = torch.zeros(N, N)  # Initialize link matrix
        self.precedence_weight = torch.zeros(N)  # Initialize precedence weight

    def write(self, w, e, a):
        """
        Write to memory.
        w: write weights
        e: erase vector
        a: add vector
        """
        self.memory = self.memory * (1 - torch.ger(w, e)) + torch.ger(w, a)

        # Update link matrix
        self.link_matrix = (1 - torch.ger(w, w) - torch.eye(self.N)) * self.link_matrix + torch.ger(w, self.precedence_weight)

        # Update precedence weight
        self.precedence_weight = (1 - w.sum()) * self.precedence_weight + w

    def read(self, w):
        """
        Read from memory.
        w: read weights
        """
        return torch.matmul(w.unsqueeze(0), self.memory).squeeze(0)

    def forward(self, read_weights, write_weights, erase_vector, add_vector):
        """
        Perform a forward pass through the memory.
        """
        self.write(write_weights, erase_vector, add_vector)
        return self.read(read_weights)
    
    def size(self):
        """
        Return the size of the memory.
        """
        return self.N, self.M