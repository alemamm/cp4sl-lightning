# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch.nn import functional as F

from .graph_layers import Diag
from .utils import apply_non_linearity, cal_similarity_graph, top_k


class FullParam(nn.Module):
    def __init__(self, n_nodes, gen_act, k, knn_metric, i):
        super().__init__()

        self.gen_act = gen_act
        self.k = k
        self.knn_metric = knn_metric
        self.i = i

        self.Adj = nn.Parameter(torch.ones(n_nodes, n_nodes))
        """if self.gen_act == "exp":

        self.Adj = nn.Parameter(
                torch.from_numpy(nearest_neighbors_pre_exp(features, self.k, self.knn_metric, self.i)))
        elif self.gen_act == "elu":
            self.Adj = nn.Parameter(
                torch.from_numpy(nearest_neighbors_pre_elu(features, self.k, self.knn_metric, self.i)))
        else:
            raise NameError('No non-linearity has been specified')
        """

    def forward(self, h):
        if self.gen_act == "exp":
            Adj = torch.exp(self.Adj)
        elif self.gen_act == "elu":
            Adj = F.elu(self.Adj) + 1
        return Adj


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
        return h

    def forward(self, features):
        embeddings = self.internal_forward(features)
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = cal_similarity_graph(embeddings)
        similarities = top_k(similarities, self.k + 1)
        similarities = apply_non_linearity(similarities, self.gen_act, self.i)
        return similarities
