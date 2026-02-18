import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from torch_geometric.data import Data


def load_feature_from_csv(f_path):
    features = pd.read_csv(f_path, header=None).values
    features = sp.csr_matrix(features, dtype=np.float32)
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    return features


def load_features(DTI_feature_path, DEN_feature_path, Chemical_feature_path):
    real_in_dim = 0
    DTIfeature = load_feature_from_csv(DTI_feature_path)
    real_in_dim += DTIfeature.shape[1]
    DENfeature = load_feature_from_csv(DEN_feature_path)
    real_in_dim += DENfeature.shape[1]
    Chemicalfeature = load_feature_from_csv(Chemical_feature_path)
    real_in_dim += Chemicalfeature.shape[1]

    in_dim = real_in_dim
    features = torch.cat((DTIfeature, DENfeature, Chemicalfeature), dim=1)

    return features, in_dim


def normalize(mx):
    mx2 = abs(mx)
    rowsum = np.array(mx2.sum(1))
    zero_mask = (rowsum == 0)
    rowsum[zero_mask] = 1e-10
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def load_csv2ddi_data(file, header=None):
    edgeweight = pd.read_csv(file, header=header, dtype='float64').values
    source = edgeweight[:, 0]
    target = edgeweight[:, 1]
    if edgeweight.shape[1] > 2:
        edgeattr = edgeweight[:, 2]
    else:
        edgeattr = np.zeros(edgeweight.shape[0])

    edgeindices = np.vstack((source, target))
    data = Data(edge_index=torch.LongTensor(edgeindices), edge_attr=torch.tensor(edgeattr))

    return data
