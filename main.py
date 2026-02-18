import numpy as np
import torch
import torch.nn as nn
from src.MKMGCN import MKMGCN_ddi_prediction
from src.utils import compute_metrics
from src.data.dataload import load_features, load_csv2ddi_data
from src.data.data import ddi_data_split
import argparse

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MKMGCN-DDI')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--task', type=str, default='type',
                        choices=['type', 'direction', 'direction-specific', 'joint-4C', 'joint-5C'])
    parser.add_argument('--q', type=float, default=0.25)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalization', type=str, default='sym')
    parser.add_argument('--average', type=str, default='macro',
                        choices=['macro', 'micro', 'weighted', 'binary'])
    parser.add_argument('--n_metrics', type=int, default=6, help='Number of metrics')
    parser.add_argument('--metrics_name', type=str, default=["AUC", "F1"])

    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    return parser.parse_args()


args = parameter_parser()

if args.task in ['type', 'direction', 'direction-specific']:
    num_classes = 2
elif args.task == 'joint-4C':
    num_classes = 4
else:  # joint-5C
    num_classes = 5

ddi_filename = "./input/dataset1/ddi_edges.csv"

target_feature_path = "input/dataset1/target_feat.csv"
enzyme_feature_path = "input/dataset1/enzyme_feat.csv"
chemical_feature_path = "input/dataset1/chemical_feat.csv"

node_feature, in_dim = load_features(target_feature_path, enzyme_feature_path, chemical_feature_path)
num_input_feat = in_dim

data = load_csv2ddi_data(ddi_filename, header=None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nepoch = args.epochs
n_metrics = args.n_metrics
metrics_name = args.metrics_name

ddi_data = ddi_data_split(data, task=args.task, prob_test=args.test_ratio, device=device)
edge_index = ddi_data['graph'].to(device)
edge_weight = ddi_data['weights'].to(device)

query_edges = ddi_data['train']['edges']
y = ddi_data['train']['labels']

query_test_edges = ddi_data['test']['edges']
y_test = ddi_data['test']['labels']

X_real = node_feature.to(device)
X_img = X_real.clone()

model = MKMGCN_ddi_prediction(q=args.q, num_features=num_input_feat, hidden=args.hidden,
                              label_dim=num_classes, dropout=args.dropout,
                              normalization=args.normalization).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
nll_loss = nn.NLLLoss()

process_data = np.zeros([nepoch, 2])


def train(X_real, X_imag, y, edge_index, edge_weight, query_edges):
    model.train()
    out, _, _ = model(X_real, X_imag, edge_index=edge_index, edge_weight=edge_weight, query_edges=query_edges)
    loss = nll_loss(out, y)
    loss.backward()
    optimizer.step()

    return loss.detach().item()


def pred(X_real, X_img, y, edge_index, edge_weight, query_edges, task):
    model.eval()
    with torch.no_grad():
        out, _, _ = model(X_real, X_img, edge_index=edge_index, edge_weight=edge_weight, query_edges=query_edges)
    test_y = y.cpu()

    metrics = compute_metrics(out, test_y, task=task, average=args.average)

    return metrics, out


print('Training...')
for epoch in range(nepoch):
    model.train()
    optimizer.zero_grad()
    train_loss = train(X_real, X_img, y, edge_index, edge_weight, query_edges)
    process_data[epoch, 0] = train_loss

    if (epoch + 1) % 10 == 0:
        print('epoch:{:03d}, Loss:{:.4f}'.format(epoch, process_data[epoch, 0]))

metrics, _ = pred(X_real, X_img, y_test, edge_index, edge_weight, query_test_edges, args.task)
metrics = metrics[: n_metrics]
print('Test results: ')
score_string = metrics_name[0] + ':{:.3f} ' + metrics_name[1] + ':{:.3f}\n '
print(score_string.format(*metrics))
