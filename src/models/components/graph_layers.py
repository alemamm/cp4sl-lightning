from typing import List

import torch
import torch.nn as nn

from .tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output)
        return output


class DenseGraphTCNConv(torch.nn.Module):
    def __init__(self, n_channels: List[int]):
        super().__init__()
        self.n_channels = n_channels
        self.tcn_rel = TCN(1, 1, num_channels=n_channels, kernel_size=7, dropout=0.0).float()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, C = x.size()
        out = torch.matmul(adj, x)
        out = self.tcn_rel(out.view(-1, C).unsqueeze(2)).view(B, N, C)
        return out
