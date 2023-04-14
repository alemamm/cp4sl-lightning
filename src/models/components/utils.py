import torch

EOS = 1e-10


def normalize_adj(adj, gen_mode):
    inv_sqrt_degree = 1.0 / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
    if gen_mode == "TCNGen":
        return inv_sqrt_degree[:, :, None] * adj * inv_sqrt_degree[:, None, :]
    else:
        return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]


def symmetrize_adj(adj, gen_mode):
    if gen_mode == "MLP_Diag" or gen_mode == "TCNGen":
        return (adj + adj.permute(0, 2, 1)) / 2
    else:
        return (adj + adj.T) / 2


def get_off_diagonal_elements(M):
    res = M.clone()
    res.diagonal(dim1=-1, dim2=-2).zero_()
    return res
