import torch
import torch.nn.functional as F

from .graph_generator import DynamicGen, StaticGen
from .graph_layers import DenseGraphTCNConv
from .utils import get_off_diagonal_elements, normalize_adj, symmetrize_adj


class GraphDAE(torch.nn.Module):
    def __init__(
        self,
        n_nodes,
        n_layers,
        gen_mode,
        dropout_dae,
        dropout_adj,
    ):
        super().__init__()
        self.dropout_dae = dropout_dae
        self.dropout_adj = dropout_adj
        self.gen_mode = gen_mode
        n_channels = [10] * 5  # [args.nhid] * args.levels for TCN within GraphTCNConv

        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(DenseGraphTCNConv(n_channels=n_channels))

        if gen_mode == "static":
            self.graph_gen = StaticGen(n_nodes)
        elif gen_mode == "dynamic":
            self.graph_gen = DynamicGen(in_dim=1, channels=[10] * 3, out_dim=1, kernel_size=7)

    def get_adj(self, h):
        Adj_, embeddings = self.graph_gen(h)
        Adj_ = symmetrize_adj(Adj_, self.gen_mode)
        Adj_ = normalize_adj(Adj_, self.gen_mode)
        Adj_ = get_off_diagonal_elements(Adj_)
        return Adj_, embeddings

    def forward(self, x, noisy_x):  # x corresponds to noisy features to be denoised
        Adj_, embeddings = self.get_adj(x)
        Adj = F.dropout(Adj_, p=self.dropout_adj)
        # Adj, Adj_ = torch.ones((8, 8)), torch.ones((8, 8)) # to try with full graph
        for i, conv in enumerate(self.layers[:-1]):
            identity = noisy_x
            noisy_x = conv(noisy_x, Adj)
            noisy_x = F.relu(noisy_x)
            noisy_x = F.dropout(noisy_x, p=self.dropout_dae)
            noisy_x = identity + noisy_x
        noisy_x = self.layers[-1](noisy_x, Adj)
        return noisy_x, Adj_, embeddings
