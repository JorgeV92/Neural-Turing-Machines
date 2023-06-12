import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def roll(t, n):
    temp = t.flip(1)
    return torch.cat((temp[:, -(n+1):], temp[:, :-(n+1)]), dim=1)