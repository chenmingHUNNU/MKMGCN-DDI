import networkx as nx
import scipy.sparse
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch_geometric
from networkx.algorithms import tree
from scipy.sparse import coo_matrix
from torch import Tensor
from torch_geometric.utils import to_undirected, negative_sampling, to_scipy_sparse_matrix


def get_ddi_samples(adj: scipy.sparse.csr_matrix,
                    edge_pairs: Tensor,
                    task: str,
                    directed_graph: bool = True,
                    typed_directed: bool = False):
    """
    This code is referenced from the
    PyTorch Geometric Signed Directed <https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/index.html>.
    """

    if edge_pairs.size(1) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    edge_pairs = np.array(edge_pairs).T

    if typed_directed:
        directed_pos = (
                np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten() > 0).tolist()
        directed_neg = (
                np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten() < 0).tolist()
        inversed_pos = (
                np.array(adj[edge_pairs[:, 1], edge_pairs[:, 0]]).flatten() > 0).tolist()
        inversed_neg = (
                np.array(adj[edge_pairs[:, 1], edge_pairs[:, 0]]).flatten() < 0).tolist()

        undirected_pos = np.logical_and(directed_pos, inversed_pos)
        undirected_neg = np.logical_and(directed_neg, inversed_neg)
        undirected_pos_neg = np.logical_and(directed_pos, inversed_neg)
        undirected_neg_pos = np.logical_and(directed_neg, inversed_pos)

        directed_pos = list(map(tuple, edge_pairs[directed_pos].tolist()))
        directed_neg = list(map(tuple, edge_pairs[directed_neg].tolist()))
        inversed_pos = list(map(tuple, edge_pairs[inversed_pos].tolist()))
        inversed_neg = list(map(tuple, edge_pairs[inversed_neg].tolist()))
        undirected = np.logical_or(np.logical_or(np.logical_or(undirected_pos, undirected_neg), undirected_pos_neg),
                                   undirected_neg_pos)
        undirected = list(map(tuple, edge_pairs[np.array(undirected)].tolist()))

        edge_pairs = list(map(tuple, edge_pairs.tolist()))

        negative = np.array(
            list(set(edge_pairs) - set(directed_pos) - set(inversed_pos) - set(directed_neg) - set(inversed_neg)))

        directed_pos = np.array(list(set(directed_pos) - set(undirected)))
        inversed_pos = np.array(list(set(inversed_pos) - set(undirected)))
        directed_neg = np.array(list(set(directed_neg) - set(undirected)))
        inversed_neg = np.array(list(set(inversed_neg) - set(undirected)))

        directed = np.vstack([directed_pos, directed_neg])
        undirected = np.array(undirected)
        new_edge_pairs = directed
        new_edge_pairs = np.vstack([new_edge_pairs, new_edge_pairs[:, [1, 0]]])
        new_edge_pairs = np.vstack([new_edge_pairs, negative])

        labels = np.vstack([np.zeros((len(directed_pos), 1), dtype=np.int32),
                            np.ones((len(directed_neg), 1), dtype=np.int32)])
        labels = np.vstack([labels, 2 * np.ones((len(directed_pos), 1), dtype=np.int32),
                            3 * np.ones((len(directed_neg), 1), dtype=np.int32)])
        labels = np.vstack(
            [labels, 4 * np.ones((len(negative), 1), dtype=np.int32)])

        edge_weights = np.vstack([np.array(adj[directed_pos[:, 0], directed_pos[:, 1]]).flatten()[:, None],
                                  np.array(adj[directed_neg[:, 0], directed_neg[:, 1]]).flatten()[:, None]])
        edge_weights = np.vstack([edge_weights, edge_weights])
        edge_weights = np.vstack(
            [edge_weights, np.zeros((len(negative), 1), dtype=np.int32)])

        assert edge_weights[labels == 0].min() > 0
        assert edge_weights[labels == 1].max() < 0
        assert edge_weights[labels == 2].min() > 0
        assert edge_weights[labels == 3].max() < 0
        assert edge_weights[labels == 4].mean() == 0


    elif directed_graph:
        directed = (np.abs(
            np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten()) > 0).tolist()
        inversed = (np.abs(
            np.array(adj[edge_pairs[:, 1], edge_pairs[:, 0]]).flatten()) > 0).tolist()
        undirected = np.logical_and(directed, inversed)

        directed = list(map(tuple, edge_pairs[directed].tolist()))
        inversed = list(map(tuple, edge_pairs[inversed].tolist()))
        undirected = list(map(tuple, edge_pairs[undirected].tolist()))

        edge_pairs = list(map(tuple, edge_pairs.tolist()))
        negative = np.array(
            list(set(edge_pairs) - set(directed) - set(inversed)))
        directed = np.array(list(set(directed) - set(undirected)))
        inversed = np.array(list(set(inversed) - set(undirected)))

        new_edge_pairs = directed
        new_edge_pairs = np.vstack([new_edge_pairs, new_edge_pairs[:, [1, 0]]])
        new_edge_pairs = np.vstack([new_edge_pairs, negative])

        labels = np.zeros((len(directed), 1), dtype=np.int32)
        labels = np.vstack([labels, np.ones((len(directed), 1), dtype=np.int32)])
        labels = np.vstack(
            [labels, 2 * np.ones((len(negative), 1), dtype=np.int32)])

        edge_weights = np.array(adj[directed[:, 0], directed[:, 1]]).flatten()[:, None]
        edge_weights = np.vstack([edge_weights, edge_weights])
        edge_weights = np.vstack(
            [edge_weights, np.zeros((len(negative), 1), dtype=np.int32)])
        assert abs(edge_weights[labels == 0]).min() > 0
        assert abs(edge_weights[labels == 1]).min() > 0
        assert edge_weights[labels == 2].mean() == 0
    else:
        undirected = []
        neg_edges = (
                np.abs(np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten()) == 0)
        labels = np.ones(len(edge_pairs), dtype=np.int32)
        labels[neg_edges] = 2
        new_edge_pairs = edge_pairs
        edge_weights = np.array(
            adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten()
        labels[edge_weights < 0] = 0
        if adj.data.min() < 0:
            assert edge_weights[labels == 0].max() < 0
        assert edge_weights[labels == 1].min() > 0
        assert edge_weights[labels == 2].mean() == 0

    if task == 'direction-specific':
        labels[labels == 1] = 0
        labels[labels == 2] = 1
        assert edge_weights[labels == 1].mean() == 0
        assert abs(edge_weights[labels == 0]).min() > 0

    return new_edge_pairs, labels.flatten(), edge_weights.flatten(), undirected


def ddi_data_split(data: torch_geometric.data.Data,
                   size: int = None,
                   prob_test: float = 0.2,
                   prob_val: float = 0,
                   task: str = 'type',
                   seed: int = 0,
                   maintain_connect: bool = True,
                   ratio: float = 1.0,
                   device: str = 'cpu') -> dict:
    """
    This code is referenced from the
    PyTorch Geometric Signed Directed <https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/index.html>.
    """
    edge_index = data.edge_index.cpu()
    row, col = edge_index[0], edge_index[1]
    if size is None:
        size = int(max(torch.max(row), torch.max(col)) + 1)

    if not hasattr(data, "edge_attr"):
        data.edge_attr = torch.ones(len(row))
    if data.edge_attr is None:
        data.edge_attr = torch.ones(len(row))

    if hasattr(data, "A"):
        A = data.A.tocsr()
    else:
        A = coo_matrix((data.edge_attr.cpu(), (row, col)),
                       shape=(size, size), dtype=np.float32).tocsr()

    len_val = int(prob_val * len(row))
    len_test = int(prob_test * len(row))

    undirect_edge_index = to_undirected(edge_index)
    if task != 'direction-specific':
        edges_neg_sam = negative_sampling(undirect_edge_index,
                                          num_neg_samples=len(edge_index.T),
                                          force_undirected=False).t().tolist()
    else:
        edges_neg_sam = negative_sampling(edge_index,
                                          num_neg_samples=len(edge_index.T),
                                          force_undirected=False).t().tolist()

    if task not in ["direction", "direction-specific"]:
        pos_ratio = (A > 0).sum() / len(A.data)
        neg_ratio = 1 - pos_ratio
        len_val_pos = int(np.around(prob_val * len(row) * pos_ratio))
        len_val_neg = int(np.around(prob_val * len(row) * neg_ratio))
        len_test_pos = int(np.around(prob_test * len(row) * pos_ratio))
        len_test_neg = int(np.around(prob_test * len(row) * neg_ratio))

    all_edge_index = edge_index.T.tolist()
    A_undirected = to_scipy_sparse_matrix(undirect_edge_index)
    if maintain_connect:
        G = nx.from_scipy_sparse_array(
            A_undirected, create_using=nx.Graph, edge_attribute='edge_attr')
        mst = list(tree.minimum_spanning_edges(
            G, algorithm="kruskal", data=False))
        all_edges = list(map(tuple, all_edge_index))
        mst_r = [t[::-1] for t in mst]
        mst += mst_r
        mst = torch.tensor(mst).t()
        nmst = list(set(all_edges) - set(mst))
    else:
        mst = []
        mst = torch.tensor(mst).t()
        nmst = edge_index.T.tolist()

    rs = np.random.RandomState(seed)
    rs.shuffle(nmst)
    rs.shuffle(edges_neg_sam)
    nmst = torch.tensor(nmst).t()
    edges_neg_sam = torch.tensor(edges_neg_sam).t()

    max_samples = int(ratio * len(edge_index.T)) + 1

    if task == 'direction':
        ids_test = torch.concat((nmst[:, : len_test], edges_neg_sam[:, : len_test]), dim=1)
        ids_val = torch.concat((nmst[:, len_test: len_test + len_val], edges_neg_sam[:, len_test: len_test + len_val]),
                               dim=1)
        if len_test + len_val < nmst.size(1):
            ids_train = torch.concat(
                (nmst[:, len_test + len_val: max_samples], mst, edges_neg_sam[:, len_test + len_val: max_samples]),
                dim=1)
        else:
            ids_train = torch.concat((mst, edges_neg_sam[:, len_test + len_val: max_samples]), dim=1)

        ids_test_directed = ids_test.clone().t().numpy()
        ids_train_directed = ids_train.clone().t().numpy()

        ids_train, labels_train, _, undirected_train = get_ddi_samples(A, ids_train, task, True)
        ids_val, labels_val, _, _ = get_ddi_samples(A, ids_val, task, True)
        ids_test, labels_test, _, _ = get_ddi_samples(A, ids_test, task, True)

    elif task == 'direction-specific':
        ids_test = torch.concat((nmst[:, : len_test], edges_neg_sam[:, : len_test]), dim=1)
        ids_val = torch.concat((nmst[:, len_test: len_test + len_val], edges_neg_sam[:, len_test: len_test + len_val]),
                               dim=1)
        if len_test + len_val < nmst.size(1):
            ids_train = torch.concat(
                (nmst[:, len_test + len_val: max_samples], mst, edges_neg_sam[:, len_test + len_val: max_samples]),
                dim=1)
        else:
            ids_train = mst + edges_neg_sam[:, len_test + len_val: max_samples]

        ids_test_directed = ids_test.clone().t().numpy()
        ids_train_directed = ids_train.clone().t().numpy()

        ids_train, labels_train, _, undirected_train = get_ddi_samples(A, ids_train, task, False)
        ids_val, labels_val, _, _ = get_ddi_samples(A, ids_val, task, False)
        ids_test, labels_test, _, _ = get_ddi_samples(A, ids_test, task, False)

    elif task == 'type':
        nmst_ = np.array(nmst).T
        pos_value_edges = torch.tensor(
            nmst_[np.array(A[nmst_[:, 0], nmst_[:, 1]] > 0).squeeze()].tolist()).t()
        neg_value_edges = torch.tensor(
            nmst_[np.array(A[nmst_[:, 0], nmst_[:, 1]] < 0).squeeze()].tolist()).t()

        ids_test = torch.concat(
            (pos_value_edges[:, : len_test_pos], neg_value_edges[:, : len_test_neg], edges_neg_sam[:, : len_test]),
            dim=1)
        ids_val = torch.concat((pos_value_edges[:, len_test_pos: len_test_pos + len_val_pos],
                                neg_value_edges[:, len_test_neg: len_test_neg + len_val_neg],
                                edges_neg_sam[:, len_test: len_test + len_val]),
                               dim=1)
        if len_test + len_val < nmst.size(1):
            ids_train = torch.concat((pos_value_edges[:, len_test_pos + len_val_pos: max_samples],
                                      neg_value_edges[:, len_test_neg + len_val_neg: max_samples],
                                      mst,
                                      edges_neg_sam[:, len_test + len_val: max_samples]), dim=1)
        else:
            ids_train = torch.concat((mst, edges_neg_sam[:, len_test + len_val: max_samples]), dim=1)

        ids_test_directed = ids_test.clone().t().numpy()
        ids_train_directed = ids_train.clone().t().numpy()

        ids_train, labels_train, _, undirected_train = get_ddi_samples(A, ids_train, task, False, False)
        ids_val, labels_val, _, _ = get_ddi_samples(A, ids_val, task, False, False)
        ids_test, labels_test, _, _ = get_ddi_samples(A, ids_test, task, False, False)

    else:
        nmst_ = np.array(nmst).T
        pos_value_edges = torch.tensor(
            nmst_[np.array(A[nmst_[:, 0], nmst_[:, 1]] > 0).squeeze()].tolist()).t()
        neg_value_edges = torch.tensor(
            nmst_[np.array(A[nmst_[:, 0], nmst_[:, 1]] < 0).squeeze()].tolist()).t()

        ids_test = torch.concat(
            (pos_value_edges[:, : len_test_pos], neg_value_edges[:, : len_test_neg], edges_neg_sam[:, : len_test]),
            dim=1)
        ids_val = torch.concat((pos_value_edges[:, len_test_pos: len_test_pos + len_val_pos],
                                neg_value_edges[:, len_test_neg: len_test_neg + len_val_neg],
                                edges_neg_sam[:, len_test: len_test + len_val]),
                               dim=1)
        if len_test + len_val < nmst.size(1):
            ids_train = torch.concat((pos_value_edges[:, len_test_pos + len_val_pos: max_samples],
                                      neg_value_edges[:, len_test_neg + len_val_neg: max_samples],
                                      mst,
                                      edges_neg_sam[:, len_test + len_val: max_samples]), dim=1)
        else:
            ids_train = torch.concat((mst, edges_neg_sam[:, len_test + len_val: max_samples]), dim=1)

        ids_test_directed = ids_test.clone().t().numpy()
        ids_train_directed = ids_train.clone().t().numpy()

        ids_train, labels_train, _, undirected_train = get_ddi_samples(A, ids_train, task, True, True)
        ids_val, labels_val, _, _ = get_ddi_samples(A, ids_val, task, True, True)
        ids_test, labels_test, _, _ = get_ddi_samples(A, ids_test, task, True, True)

    if task in ['type', 'direction']:
        ids_train = ids_train[labels_train < 2]
        # label_train_w = label_train_w[labels_train <2]
        labels_train = labels_train[labels_train < 2]

        ids_test = ids_test[labels_test < 2]
        # label_test_w = label_test_w[labels_test <2]
        labels_test = labels_test[labels_test < 2]


    elif task == 'joint-4C':
        ids_train = ids_train[labels_train < 4]
        labels_train = labels_train[labels_train < 4]
        # weights_train = weights_train[labels_train < 4]

        ids_test = ids_test[labels_test < 4]
        labels_test = labels_test[labels_test < 4]
        # weights_test = weights_test[labels_test < 4]

    if len(ids_train) > 0:

        if task == 'type':
            edge_attr = data.edge_attr.cpu()
            edge_index_undirected, edge_attr_undirected = to_undirected(edge_index, edge_attr)
            row, col = edge_index_undirected[0], edge_index_undirected[1]
            size = int(max(torch.max(row), torch.max(col)) + 1)
            A = coo_matrix((edge_attr_undirected, (row, col)), shape=(size, size), dtype=np.float32).tocsr()

            ids_train_undirected = to_undirected(torch.from_numpy(ids_train).t()).t().numpy()
            observed_edges = -np.ones((len(ids_train_undirected), 2), dtype=np.int32)
            observed_weight = np.zeros((len(ids_train_undirected), 1), dtype=np.float32)

            direct = (
                    np.abs(A[ids_train_undirected[:, 0], ids_train_undirected[:, 1]].data) > 0).flatten()
            observed_edges[direct, 0] = ids_train_undirected[direct, 0]
            observed_edges[direct, 1] = ids_train_undirected[direct, 1]
            observed_weight[direct, 0] = np.array(
                A[ids_train_undirected[direct, 0], ids_train_undirected[direct, 1]]).flatten()

        else:
            observed_edges = -np.ones((len(ids_train_directed), 2), dtype=np.int32)
            observed_weight = np.zeros((len(ids_train_directed), 1), dtype=np.float32)

            direct = (
                    np.abs(A[ids_train_directed[:, 0], ids_train_directed[:, 1]].data) > 0).flatten()
            observed_edges[direct, 0] = ids_train_directed[direct, 0]
            observed_edges[direct, 1] = ids_train_directed[direct, 1]
            observed_weight[direct, 0] = np.array(
                A[ids_train_directed[direct, 0], ids_train_directed[direct, 1]]).flatten()

        valid = (np.sum(observed_edges, axis=-1) >= 0)
        observed_edges = observed_edges[valid]
        observed_weight = observed_weight[valid]

        if len(undirected_train) > 0:
            undirected_train = np.array(undirected_train)
            observed_edges = np.vstack(
                (observed_edges, undirected_train))
            observed_weight = np.vstack((observed_weight, np.array(A[undirected_train[:, 0],
            undirected_train[:, 1]]).flatten()[:, None]))

        if task != 'type':
            assert (len(edge_index.T) >= len(observed_edges)), 'The original edge number is {} \
                while the observed graph has {} edges!'.format(len(edge_index.T), len(observed_edges))
    else:
        observed_edges = np.array([])
        observed_weight = np.array([])

    datasets = {}
    datasets['graph'] = torch.from_numpy(observed_edges.T).long().to(device)
    datasets['weights'] = torch.from_numpy(observed_weight.flatten()).float().to(device)

    datasets['train'] = {}
    datasets['train']['edges'] = torch.from_numpy(ids_train).long().to(device)
    datasets['train']['labels'] = torch.from_numpy(labels_train).long().to(device)
    # datasets[ind]['train']['weight'] = torch.from_numpy(label_train_w).float().to(device)

    datasets['val'] = {}
    datasets['val']['edges'] = torch.from_numpy(ids_val).long().to(device)
    datasets['val']['labels'] = torch.from_numpy(labels_val).long().to(device)
    # datasets[ind]['val']['weight'] = torch.from_numpy(label_val_w).float().to(device)

    datasets['test'] = {}
    datasets['test']['edges'] = torch.from_numpy(ids_test).long().to(device)
    datasets['test']['labels'] = torch.from_numpy(labels_test).long().to(device)
    # datasets[ind]['test']['weight'] = torch.from_numpy(label_test_w).float().to(device)

    return datasets
