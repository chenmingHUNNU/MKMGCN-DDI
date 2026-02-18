import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from typing import Optional
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from .get_adj_norm import get_adj_norm


class MKMConv(MessagePassing):
    """
    The convolutional layer of MKMGCN-DDI.
    This code is referenced from the
    PyTorch Geometric Signed Directed <https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/index.html>.
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 q: float,
                 K: int = 2,
                 normalization: str = 'sym',
                 cached: bool = False,
                 bias: bool = True,
                 first_aggr: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MKMConv, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym'], 'Invalid normalization'
        kwargs.setdefault('flow', 'target_to_source')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.cached = cached
        self.q = q

        if first_aggr:
            self.weight_pos = Parameter(Tensor(K, in_channels, out_channels))
            self.weight_neg = Parameter(Tensor(K, in_channels, out_channels))

        else:
            self.weight_pos = Parameter(Tensor(K, 4 * in_channels, out_channels))
            self.weight_neg = Parameter(Tensor(K, 4 * in_channels, out_channels))

        if bias:
            self.bias = Parameter(Tensor(4 * out_channels))
        else:
            self.register_parameter('bias', None)

        self.act_func = nn.LeakyReLU()

        self.pos_att_src = Parameter(torch.empty(2, 1, out_channels))
        self.pos_att_dst = Parameter(torch.empty(2, 1, out_channels))
        self.neg_att_src = Parameter(torch.empty(2, 1, out_channels))
        self.neg_att_dst = Parameter(torch.empty(2, 1, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_pos)
        glorot(self.weight_neg)
        zeros(self.bias)

        self.cached_result = None
        self.cached_num_edges = None
        self.cached_q = None

    def __norm__(
            self,
            edge_index,
            num_nodes: Optional[int],
            edge_weight: OptTensor,
            q: float,
            mode,
            dtype: Optional[int] = None
    ):
        edge_index, edge_weight_real, edge_weight_imag = get_adj_norm(
            edge_index, edge_weight, dtype, num_nodes, q, mode
        )

        edge_weight_real.masked_fill_(edge_weight_real == float("inf"), 0)
        edge_weight_imag.masked_fill_(edge_weight_imag == float("inf"), 0)

        edge_index_real = edge_index_imag = edge_index.clone()

        return edge_index_real, edge_index_imag, edge_weight_real, edge_weight_imag

    def forward(self, x_real: torch.FloatTensor, x_imag: torch.FloatTensor,
                edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None):

        pos_edge_index = edge_index[:, edge_weight > 0]
        pos_edge_weight = edge_weight[edge_weight > 0]
        neg_edge_index = edge_index[:, edge_weight < 0]
        neg_edge_weight = - edge_weight[edge_weight < 0]

        pos_l_edge_index_real, pos_l_edge_index_imag, pos_l_norm_real, pos_l_norm_imag = self.__norm__(
            pos_edge_index, x_real.size(self.node_dim), pos_edge_weight, self.q, mode='low', dtype=x_real.dtype)
        pos_h_edge_index_real, pos_h_edge_index_imag, pos_h_norm_real, pos_h_norm_imag = self.__norm__(
            pos_edge_index, x_real.size(self.node_dim), pos_edge_weight, self.q, mode='high', dtype=x_real.dtype)

        neg_l_edge_index_real, neg_l_edge_index_imag, neg_l_norm_real, neg_l_norm_imag = self.__norm__(
            neg_edge_index, x_real.size(self.node_dim), neg_edge_weight, self.q, mode='low', dtype=x_real.dtype)
        neg_h_edge_index_real, neg_h_edge_index_imag, neg_h_norm_real, neg_h_norm_imag = self.__norm__(
            neg_edge_index, x_real.size(self.node_dim), neg_edge_weight, self.q, mode='high', dtype=x_real.dtype)

        pos_l_out_real, pos_l_out_imag = self.compute_message_aggregation(
            x_real, x_imag, pos_l_edge_index_real, pos_l_edge_index_imag,
            pos_l_norm_real, pos_l_norm_imag, weight=self.weight_pos)
        pos_h_out_real, pos_h_out_imag = self.compute_message_aggregation(
            x_real, x_imag, pos_h_edge_index_real, pos_h_edge_index_imag,
            pos_h_norm_real, pos_h_norm_imag, weight=self.weight_pos)

        neg_l_out_real, neg_l_out_imag = self.compute_message_aggregation(
            x_real, x_imag, neg_l_edge_index_real, neg_l_edge_index_imag,
            neg_l_norm_real, neg_l_norm_imag, weight=self.weight_neg)
        neg_h_out_real, neg_h_out_imag = self.compute_message_aggregation(
            x_real, x_imag, neg_h_edge_index_real, neg_h_edge_index_imag,
            neg_h_norm_real, neg_h_norm_imag, weight=self.weight_neg)

        out_real = torch.cat((pos_l_out_real, pos_h_out_real, neg_l_out_real, neg_h_out_real), dim=-1)
        out_imag = torch.cat((pos_l_out_imag, pos_h_out_imag, neg_l_out_imag, neg_h_out_imag), dim=-1)

        if self.bias is not None:
            out_real += self.bias
            out_imag += self.bias

        out_real = self.act_func(out_real)
        out_imag = self.act_func(out_imag)

        return out_real, out_imag

    def compute_message_aggregation(self,
                                    x_real, x_imag,
                                    edge_index_real, edge_index_imag,
                                    norm_real, norm_imag,
                                    weight):
        Tx_0_real_real = x_real
        Tx_0_imag_imag = x_imag

        Tx_0_out_real_real = torch.matmul(Tx_0_real_real, weight[0])
        Tx_0_out_imag_imag = torch.matmul(Tx_0_imag_imag, weight[0])

        Tx_1_real_real = torch.matmul(x_real, weight[1])
        Tx_1_out_real_real = self.propagate(
            edge_index_real, x=Tx_1_real_real, norm=norm_real, size=None)

        Tx_1_imag_imag = torch.matmul(x_imag, weight[1])
        Tx_1_out_imag_imag = self.propagate(
            edge_index_imag, x=Tx_1_imag_imag, norm=norm_imag, size=None)

        Tx_1_imag_real = torch.matmul(x_imag, weight[1])
        Tx_1_out_imag_real = self.propagate(
            edge_index_real, x=Tx_1_imag_real, norm=norm_real, size=None)

        Tx_1_real_imag = torch.matmul(x_real, weight[1])
        Tx_1_out_real_imag = self.propagate(
            edge_index_imag, x=Tx_1_real_imag, norm=norm_imag, size=None)

        out_real = Tx_0_out_real_real + Tx_1_out_real_real - Tx_1_out_imag_imag
        out_imag = Tx_0_out_imag_imag + Tx_1_out_imag_real + Tx_1_out_real_imag

        return out_real, out_imag

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight_pos.size(0), self.normalization)
