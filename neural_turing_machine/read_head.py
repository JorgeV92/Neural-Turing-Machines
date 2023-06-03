import torch
import torch.nn.functional as F
from torch import nn

class ReadHead(nn.Module):
    def __init__(self, memory_size, memory_vector_dim):
        super(ReadHead, self).__init__()
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.key_dim = memory_vector_dim

        # Define the linear layer to generate the key
        self.key_layer = nn.Linear(self.key_dim, self.key_dim)

    def forward(self, controller_output, prev_weights, memory):
        # Compute the key from the controller output
        key = self.key_layer(controller_output)
        
        # Normalizing the key (optional)
        key = F.normalize(key)

        # Compute the cosine similarity
        similarity_scores = F.cosine_similarity(memory + 1e-16, key.unsqueeze(0) + 1e-16, dim=-1)
        
        # Use softmax to compute the weights from the similarity scores
        weights = F.softmax(similarity_scores, dim=-1)
        
        # Retrieve the memory
        read_vector = torch.sum(memory * weights.unsqueeze(2), dim=1)
        
        return read_vector, weights
