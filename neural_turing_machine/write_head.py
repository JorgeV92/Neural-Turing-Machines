import torch
import torch.nn.functional as F
from torch import nn

class WriteHead(nn.Module):
    def __init__(self, memory_size, memory_vector_dim):
        super(WriteHead, self).__init__()
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.key_dim = memory_vector_dim

        # Define the linear layers to generate the key, erase, and add vectors
        self.key_layer = nn.Linear(self.key_dim, self.key_dim)
        self.erase_layer = nn.Linear(self.key_dim, self.memory_vector_dim)
        self.add_layer = nn.Linear(self.key_dim, self.memory_vector_dim)

    def forward(self, controller_output, prev_weights, memory):
        # Compute the key, erase, and add vectors from the controller output
        key = self.key_layer(controller_output)
        erase_vector = torch.sigmoid(self.erase_layer(controller_output))
        add_vector = self.add_layer(controller_output)

        # Normalizing the key (optional)
        key = F.normalize(key)

        # Compute the cosine similarity
        similarity_scores = F.cosine_similarity(memory + 1e-16, key.unsqueeze(0) + 1e-16, dim=-1)

        # Use softmax to compute the weights from the similarity scores
        weights = F.softmax(similarity_scores, dim=-1)

        # Erase and then add to the memory
        memory = memory * (1 - weights.unsqueeze(2) * erase_vector.unsqueeze(1)) + weights.unsqueeze(2) * add_vector.unsqueeze(1)

        return memory, weights
