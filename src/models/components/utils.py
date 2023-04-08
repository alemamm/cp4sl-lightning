import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph

EOS = 1e-10


def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == "elu":
        return F.elu(tensor * i - i) + 1
    elif non_linearity == "relu":
        return F.relu(tensor)
    elif non_linearity == "none":
        return tensor
    else:
        raise NameError("We dont support the non-linearity yet")


def get_random_mask(features, r, nr):
    nones = torch.sum(features > 0.0).float()
    nzeros = features.shape[0] * features.shape[1] - nones
    pzeros = nones / nzeros / r * nr
    probs = torch.zeros(features.shape)
    probs[features == 0.0] = pzeros
    probs[features > 0.0] = 1 / r
    mask = torch.bernoulli(probs)
    return mask


def accuracy(preds, labels):
    pred_class = torch.max(preds, 1)[1]
    return torch.sum(torch.eq(pred_class, labels)).float() / labels.shape[0]


def nearest_neighbors(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj


def nearest_neighbors_pre_exp(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def normalize_adj(adj, mode, gen_mode):
    if mode == "sym":
        if gen_mode == "MLP_Diag" or gen_mode == "TCNGen":
            inv_sqrt_degree = 1.0 / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            sym = inv_sqrt_degree[:, :, None] * adj * inv_sqrt_degree[:, None, :]
            # print("sym dims", sym.shape)
            return sym
            # deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
            # sym = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
            # return sym
        else:
            inv_sqrt_degree = 1.0 / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
    elif mode == "row":
        inv_degree = 1.0 / (adj.sum(dim=1, keepdim=False) + EOS)
        return inv_degree[:, None] * adj
    else:
        exit("wrong norm mode")


def symmetrize_adj(adj, gen_mode):
    if gen_mode == "MLP_Diag" or gen_mode == "TCNGen":
        return (adj + adj.permute(0, 2, 1)) / 2
    else:
        return (adj + adj.T) / 2


def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.bmm(node_embeddings, node_embeddings.permute(0, 2, 1))
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[2]
    mask = torch.zeros(raw_graph.shape)

    print("mask", mask.shape, "raw_graph", raw_graph.shape)

    mask[torch.arange(raw_graph.shape[1]).view(-1, 1), indices] = 1.0

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph


def tril_values(x):
    if len(x.shape) == 2:
        mask = torch.ones(x.shape[0], x.shape[0])
        return x[mask.triu(diagonal=1) == 1]
    elif len(x.shape) == 3:
        mask = torch.ones(x.shape[1], x.shape[1])
        return x[:, mask.triu(diagonal=1) == 1]
    else:
        raise ValueError(f"cannot deal with this shape {x.shape}")


def get_off_diagonal_elements(M):
    res = M.clone()
    res.diagonal(dim1=-1, dim2=-2).zero_()
    return res
