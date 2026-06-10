"""
The seven metrics reported in the paper's result tables (Table 1, A.2-A.5).

We keep them in one function so every model in the project is scored the
exact same way -- that is the only fair way to compare tabular ML, node2vec
and the GNN against each other.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    recall_score,
    precision_score,
)


def evaluate(y_true, y_prob, threshold=0.5):
    """Score one model.

    y_prob is the predicted probability of the positive class (investment
    grade = 1). We threshold it at 0.5 to get hard labels for the metrics
    that need them, but ROC-AUC uses the raw probability.

    'Mean recall' in the paper is the macro-averaged recall (the unweighted
    average of recall on each class), which rewards a model for doing well on
    the smaller class too -- important here because the data is imbalanced.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "F1": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Mean recall": recall_score(y_true, y_pred, average="macro"),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }


def aggregate(rows):
    """Average a list of metric dicts across replications and report the
    standard deviation, mirroring how the paper prints 'mean (std)'."""
    keys = rows[0].keys()
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    std = {k: float(np.std([r[k] for r in rows])) for k in keys}
    return mean, std
