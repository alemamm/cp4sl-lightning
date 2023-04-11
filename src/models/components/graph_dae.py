import torch
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseGCNConv

from .graph_generator import FullParam, MLP_Diag, TCNGen
from .graph_layers import DenseGraphConv, DenseGraphTCNConv, DenseGraphTemporalConv
from .utils import get_off_diagonal_elements, normalize_adj, symmetrize_adj


class GraphDAE(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_nodes,
        n_layers,
        gen_mode,
        gen_act,
        k,
        knn_metric,
        i_,
        normalization,
        dropout_dae,
        dropout_adj,
        graph_layer,
    ):
        super().__init__()
        self.normalization = normalization
        self.dropout_dae = dropout_dae
        self.dropout_adj = dropout_adj
        self.gen_mode = gen_mode

        n_channels = [10] * 5  # [args.nhid] * args.levels

        out_dim = in_dim  # input and output have same dimensions since we are doing denoising/reconstruction

        self.layers = torch.nn.ModuleList()

        if graph_layer == "GraphConv":
            self.layers.append(DenseGraphConv(in_dim, hidden_dim, aggr="add"))
            for _ in range(n_layers - 2):
                self.layers.append(DenseGraphConv(in_dim, hidden_dim, aggr="add"))
            self.layers.append(DenseGraphConv(hidden_dim, out_dim, aggr="add"))

        elif graph_layer == "GCNConv":
            self.layers.append(DenseGCNConv(in_dim, hidden_dim))
            for _ in range(n_layers - 2):
                self.layers.append(DenseGCNConv(hidden_dim, hidden_dim))
            self.layers.append(DenseGCNConv(hidden_dim, out_dim))

        elif graph_layer == "GraphTemporalConv":
            self.layers.append(
                DenseGraphTemporalConv(in_dim, hidden_dim, aggr="add", dilation_size=2)
            )
            self.layers.append(
                DenseGraphTemporalConv(hidden_dim, hidden_dim, aggr="add", dilation_size=4)
            )
            self.layers.append(
                DenseGraphTemporalConv(hidden_dim, out_dim, aggr="add", dilation_size=8)
            )
            self.layers.append(torch.nn.Linear(out_dim, out_dim))

        elif graph_layer == "GraphTCNConv":
            self.layers.append(DenseGraphTCNConv(n_channels=n_channels, aggr="mean"))
            for _ in range(n_layers - 2):
                self.layers.append(DenseGraphTCNConv(n_channels=n_channels, aggr="mean"))
            self.layers.append(DenseGraphTCNConv(n_channels=n_channels, aggr="mean"))

        if gen_mode == "FP":
            self.graph_gen = FullParam(
                n_nodes,
                gen_act,
                k,
                knn_metric,
                i_,
            )

        elif gen_mode == "MLP_Diag":
            self.graph_gen = MLP_Diag(
                2, 200, k, knn_metric, gen_act, i_
            )  # make number of layers a parameter?

        elif gen_mode == "TCNGen":
            self.graph_gen = TCNGen(
                in_dim=1,
                channels=[10] * 3,
                out_dim=1,
                kernel_size=7,
                k=k,
                knn_metric=knn_metric,
                i=i_,
                gen_act=gen_act,
            )

    def get_adj(self, h):
        Adj_ = self.graph_gen(h)
        Adj_ = symmetrize_adj(Adj_, self.gen_mode)
        Adj_ = normalize_adj(Adj_, self.normalization, self.gen_mode)
        Adj_ = get_off_diagonal_elements(Adj_)
        return Adj_

    def forward(self, x, noisy_x):  # x corresponds to noisy features to be denoised
        Adj_ = self.get_adj(x)
        Adj = F.dropout(Adj_, p=self.dropout_adj)
        # Adj, Adj_ = torch.ones((8, 8)), torch.ones((8, 8))
        identity = noisy_x
        for i, conv in enumerate(self.layers[:-1]):
            # identity = noisy_x
            noisy_x = conv(noisy_x, Adj)  # , add_loop=False)
            # noisy_x = noisy_x + identity
            noisy_x = F.relu(noisy_x)
            noisy_x = F.dropout(noisy_x, p=self.dropout_dae)
            noisy_x = identity + noisy_x  # make residual connection option?
        # add option to add residual connection to last layer
        noisy_x = self.layers[-1](noisy_x, Adj)  # , add_loop=False)  # no dropout on last layer
        return noisy_x, Adj_
