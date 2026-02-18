from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.MKMGCN import MKMConv


class MKMGCN_ddi_prediction(nn.Module):
    """
    The MKMGCN-DDI model for DDIs prediction
    This code is referenced from the
    PyTorch Geometric Signed Directed <https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/index.html>
    """

    def __init__(self, num_features: int, hidden: int = 2, q: float = 0.25, K: int = 2, label_dim: int = 2,
                 layer: int = 2, dropout: float = 0.5,
                 normalization: str = 'sym',
                 cached: bool = False,
                 conv_bias: bool = True
                 ):
        super(MKMGCN_ddi_prediction, self).__init__()
        chebs = nn.ModuleList()
        chebs.append(
            MKMConv(in_channels=num_features, out_channels=hidden, K=K, q=q,
                    normalization=normalization, bias=conv_bias, first_aggr=True))
        self.normalization = normalization

        self.act_func = torch.nn.ReLU()

        for _ in range(1, layer):
            chebs.append(
                MKMConv(in_channels=hidden, out_channels=hidden, K=K, q=q,
                        normalization=normalization, bias=conv_bias))

        self.Chebs = chebs
        self.linear1 = nn.Linear(hidden * 4 * 4, hidden * 2)
        self.linear2 = nn.Linear(hidden * 2, label_dim)
        self.dropout = dropout
        MKMGCN_ddi_prediction.reset_parameters(self)

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor,
                query_edges: torch.LongTensor, edge_weight: Optional[torch.LongTensor] = None):
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)

        x = torch.cat(
            (real[query_edges[:, 0]], real[query_edges[:, 1]], imag[query_edges[:, 0]], imag[query_edges[:, 1]]),
            dim=-1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.linear1(x)
        x = self.act_func(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)

        return x, real, imag
