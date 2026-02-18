from typing import Optional
import torch
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import add_self_loops, remove_self_loops, to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np


def get_adj_norm(edge_index: torch.LongTensor,
                 edge_weight: Optional[torch.Tensor] = None,
                 dtype: Optional[int] = None,
                 num_nodes: Optional[int] = None,
                 q: Optional[float] = 0.25,
                 mode='low'):
    """
    This code is referenced from the
    PyTorch Geometric Signed Directed <https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/index.html>.
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    theta_attr = torch.cat([edge_weight, -edge_weight], dim=0)
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)
    edge_attr = torch.stack([sym_attr, theta_attr], dim=1)

    edge_index_sym, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                         num_nodes, "mean")

    edge_weight_sym = edge_attr[:, 0]
    edge_weight_sym = edge_weight_sym / 2
    row, col = edge_index_sym[0], edge_index_sym[1]
    deg = scatter_add(edge_weight_sym, row, dim=0, dim_size=num_nodes)

    Theta = 1j * 2 * np.pi * q * edge_attr[:, 1]
    edge_weight_q = torch.exp(Theta)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * \
                  edge_weight_sym * deg_inv_sqrt[col] * edge_weight_q

    if mode == 'low':
        return edge_index_sym, edge_weight.real, edge_weight.imag

    else:
        return edge_index_sym, - edge_weight.real, - edge_weight.imag
