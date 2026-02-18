import numpy as np
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (roc_auc_score, f1_score,
                             average_precision_score,
                             precision_score, recall_score,
                             accuracy_score)


def compute_metrics(out, y, n_class=None, task='type', average='macro', multi_class='ovo'):

    pred = out.max(dim=1)[1].detach().cpu().numpy()
    y = y.numpy()

    acc = accuracy_score(y, pred)

    if task == 'direction-specific':
        precision = precision_score(y, pred, average='binary', pos_label=0, zero_division=0)
        recall = recall_score(y, pred, average='binary', pos_label=0, zero_division=0)
        f1 = f1_score(y, pred, average='binary', pos_label=0, zero_division=0) if pred.sum() > 0 else 0

        out_exp = torch.exp(out).detach().cpu().numpy()
        y_score = out_exp[:, 1]
        auc = roc_auc_score(y, y_score)
        y_score = out_exp[:, 0]
        ap = average_precision_score(y, y_score, pos_label=0)

    elif task in ['type', 'direction']:
        precision = precision_score(y, pred, average=average, zero_division=0)
        recall = recall_score(y, pred, average=average, zero_division=0)
        f1 = f1_score(y, pred, average=average, zero_division=0)
        f1_micro = f1_score(y, pred, average='micro')

        out_exp = torch.exp(out).detach().cpu().numpy()
        y_score = out_exp[:, 1]
        auc = roc_auc_score(y, y_score)
        y_score = out_exp[:, 0]
        ap = average_precision_score(y, y_score, pos_label=0)

    else:
        precision = precision_score(y, pred, average=average, zero_division=0)
        recall = recall_score(y, pred, average=average, zero_division=0)
        f1 = f1_score(y, pred, average=average, zero_division=0)

        classes = np.unique(y)
        y_binarized = label_binarize(y, classes=classes)
        out_exp = torch.exp(out).detach().cpu().numpy()
        auc = roc_auc_score(y_binarized, out_exp, average=average, multi_class=multi_class)
        ap = average_precision_score(y_binarized, out_exp, average=average)

    return auc, f1