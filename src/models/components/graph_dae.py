import torch
import torch.nn.functional as F

from .graph_generator import DynamicGen, StaticGen
from .graph_layers import DenseGraphTCNConv
from .utils import get_off_diagonal_elements, normalize_adj, symmetrize_adj


class GraphDAE(torch.nn.Module):
    def __init__(
        self,
        n_nodes,  # number of nodes in the graph
        n_layers,  # layers of DAE
        gen_mode,  # "static" or "dynamic"
        n_channels,  # [args.nhid] * args.levels for TCN within GraphTCNConv
        dropout_adj,  # dropout for graph generator output
        dropout_dae,  # dropout for hidden DAE layers
        dropout_graphtcn,  # dropout for TCN layers within GraphTCNConv
        kernel_size,  # kernel size for TCN
        seed,  # random seed
        features_as_embeddings,  # True to test importance of dynamic generator
        use_full_graph,  # True to use full graph to test importance of graph generator module
        use_true_adj,  # True to use true adjacency matrix as input to DAE
        use_correlation_matrix,  # True to use correlation matrix as input to DAE
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.dropout_adj = dropout_adj
        self.dropout_dae = dropout_dae
        self.dropout_graphtcn = dropout_graphtcn
        self.gen_mode = gen_mode
        self.kernel_size = kernel_size
        self.seed = seed
        self.features_as_embeddings = features_as_embeddings
        self.use_full_graph = use_full_graph
        self.full_graph_adj = torch.ones((self.n_nodes, self.n_nodes), device="cuda:0")
        self.use_true_adj = use_true_adj
        # uncomment line below to run true graph ablations once data has been created in prior "normal run"
        # self.true_adj = torch.load(f"data/kuramoto/{seed}_2_4/normal_adj.pt").to("cuda:0") # run once created
        self.use_correlation_matrix = use_correlation_matrix
        # uncomment line below to run correlation ablations once correlations created using notebook
        # self.correlation_matrix = torch.load(f"data/kuramoto/{seed}_2_4/correlation_matrix.pt").to("cuda:0")

        self.n_n_channels = n_channels = [10, 10, 10, 10, 10]

        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                DenseGraphTCNConv(
                    n_channels=n_channels,
                    kernel_size=self.kernel_size,
                    dropout=self.dropout_graphtcn,
                )
            )

        if gen_mode == "static":
            self.graph_gen = StaticGen(n_nodes)
        elif gen_mode == "dynamic":
            self.graph_gen = DynamicGen(
                in_dim=1, channels=[10, 10, 10], out_dim=1, kernel_size=self.kernel_size
            )

    def get_adj(self, h):
        Adj_, embeddings = self.graph_gen(h, self.features_as_embeddings)
        Adj_ = symmetrize_adj(Adj_, self.gen_mode)
        Adj_ = normalize_adj(Adj_, self.gen_mode)
        Adj_ = get_off_diagonal_elements(Adj_)
        return Adj_, embeddings

    def forward(self, x, noisy_x):  # x corresponds to noisy features to be denoised
        Adj_, embeddings = self.get_adj(x)
        Adj = F.dropout(Adj_, p=self.dropout_adj)
        # used to test the importance of the graph generator module
        if self.use_full_graph:
            Adj = get_off_diagonal_elements(self.full_graph_adj)
            Adj_ = get_off_diagonal_elements(self.full_graph_adj)
        # used to evaluate reconstruction performance if the true adjacency matrix is known
        if self.use_true_adj:
            Adj = get_off_diagonal_elements(self.true_adj)
            Adj_ = get_off_diagonal_elements(self.true_adj)
        # used to evaluate reconstruction performance if the correlation matrix is used instead of graph generator
        if self.use_correlation_matrix:
            Adj = get_off_diagonal_elements(self.correlation_matrix)
            Adj_ = get_off_diagonal_elements(self.correlation_matrix)
        for i, conv in enumerate(self.layers[:-1]):
            identity = noisy_x
            noisy_x = conv(noisy_x, Adj)
            noisy_x = F.relu(noisy_x)
            noisy_x = F.dropout(noisy_x, p=self.dropout_dae)
            noisy_x = identity + noisy_x
        noisy_x = self.layers[-1](noisy_x, Adj)
        return noisy_x, Adj_, embeddings
