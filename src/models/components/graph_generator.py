# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch.nn import functional as F

from .graph_layers import TCN, Diag
from .utils import apply_non_linearity, cal_similarity_graph, top_k


class FullParam(nn.Module):
    def __init__(self, n_nodes, gen_act, k, knn_metric, i):
        super().__init__()

        self.gen_act = gen_act

        self.Adj = nn.Parameter(torch.ones(n_nodes, n_nodes))

    def forward(self, h):
        # if self.gen_act == "exp":
        #    Adj = torch.exp(self.Adj)
        # elif self.gen_act == "elu":
        #    Adj = F.elu(self.Adj) + 1
        return F.elu(self.Adj)


class MLP_Diag(nn.Module):
    def __init__(self, nlayers, n_timesteps, k, knn_metric, gen_act, i):
        super().__init__()

        self.i = i
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Diag(n_timesteps))
        self.k = k
        self.knn_metric = knn_metric
        self.gen_act = gen_act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.gen_act == "relu" or "elu":
                    h = F.relu(h)
                elif self.gen_act == "tanh":
                    h = torch.tanh(h)
                elif self.gen_act == "sigmoid":
                    h = torch.sigmoid(h)
        return h

    def forward(self, features):
        embeddings = self.internal_forward(features)
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = cal_similarity_graph(embeddings)
        # similarities = top_k(similarities, self.k + 1)
        # similarities = apply_non_linearity(similarities, self.gen_act, self.i)
        return similarities


class TCNGen(nn.Module):
    def __init__(self, in_dim, channels, out_dim, kernel_size, k, knn_metric, i, gen_act):
        super().__init__()
        self.tcn = TCN(in_dim, out_dim, channels, kernel_size, dropout=0.3)
        self.i = i
        self.k = k
        self.knn_metric = knn_metric
        self.gen_act = gen_act

    def internal_forward(self, h):
        B, N, C = h.size()
        h = self.tcn(h.view(-1, C).unsqueeze(2)).view(B, N, C)
        return torch.sigmoid(h)

    def forward(self, features):
        embeddings = self.internal_forward(features)
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = cal_similarity_graph(embeddings)
        # similarities = top_k(similarities, self.k + 1)
        # similarities = apply_non_linearity(similarities, self.gen_act, self.i)
        return similarities
