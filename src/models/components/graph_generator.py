import torch
import torch.nn as nn
from torch.nn import functional as F

from .graph_layers import TCN


class StaticGen(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        self.Adj = nn.Parameter(torch.ones(n_nodes, n_nodes))

    def forward(self, h):
        embeddings = None
        return F.elu(self.Adj), embeddings


class DynamicGen(nn.Module):
    def __init__(self, in_dim, channels, out_dim, kernel_size):
        super().__init__()
        self.tcn = TCN(in_dim, out_dim, channels, kernel_size, dropout=0.3)

    def internal_forward(self, h):
        B, N, C = h.size()
        h = self.tcn(h.view(-1, C).unsqueeze(2)).view(B, N, C)
        return torch.sigmoid(h)

    def forward(self, features):
        embeddings = self.internal_forward(features)
        # embeddings = torch.sigmoid(features) # use to test the the importance of the TCN generator
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = torch.bmm(embeddings, embeddings.permute(0, 2, 1))
        return similarities, embeddings
