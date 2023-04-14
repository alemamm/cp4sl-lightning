from typing import List, Optional

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
        output = self.linear(output)  # .double()
        return output


class DenseGraphTCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GraphConv`."""

    def __init__(
        self,
        n_channels: List[int],
        aggr: str = "add",
    ):
        assert aggr in ["add", "mean", "max"]
        super().__init__()
        self.n_channels = n_channels
        self.aggr = aggr

        self.tcn_rel = TCN(1, 1, num_channels=n_channels, kernel_size=7, dropout=0.3).float()

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, C = x.size()

        if self.aggr == "add":
            out = torch.matmul(adj, x)
        elif self.aggr == "mean":
            out = torch.matmul(adj, x)
            out = out / adj.sum(dim=-1, keepdim=True).clamp_(min=1)
        elif self.aggr == "max":
            out = x.unsqueeze(-2).repeat(1, 1, N, 1)
            adj = adj.unsqueeze(-1).expand(B, N, N, C)
            out[adj == 0] = float("-inf")
            out = out.max(dim=-3)[0]
            out[out == float("-inf")] = 0.0
        else:
            raise NotImplementedError

        out = self.tcn_rel(out.view(-1, C).unsqueeze(2)).view(B, N, C)

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.n_channels})"


class DenseGraphConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GraphConv`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "add",
        bias: bool = True,
    ):
        assert aggr in ["add", "mean", "max"]
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.lin_rel = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin_rel.reset_parameters()

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, C = x.size()

        if self.aggr == "add":
            out = torch.matmul(adj, x)
        elif self.aggr == "mean":
            out = torch.matmul(adj, x)
            out = out / adj.sum(dim=-1, keepdim=True).clamp_(min=1)
        elif self.aggr == "max":
            out = x.unsqueeze(-2).repeat(1, 1, N, 1)
            adj = adj.unsqueeze(-1).expand(B, N, N, C)
            out[adj == 0] = float("-inf")
            out = out.max(dim=-3)[0]
            out[out == float("-inf")] = 0.0
        else:
            raise NotImplementedError

        out = self.lin_rel(out)

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, " f"{self.out_channels})"
