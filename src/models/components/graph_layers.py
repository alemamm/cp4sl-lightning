from typing import List

import torch

from .tcn import TCN


class DenseGraphTCNConv(torch.nn.Module):
    def __init__(self, n_channels: List[int], kernel_size: int = 15, dropout: float = 0.3):
        super().__init__()
        self.n_channels = n_channels
        self.tcn_rel = TCN(
            1, 1, num_channels=n_channels, kernel_size=kernel_size, dropout=dropout
        ).float()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, C = x.size()
        out = torch.matmul(adj, x)
        out = self.tcn_rel(out.view(-1, C).unsqueeze(2)).view(B, N, C)
        return out
