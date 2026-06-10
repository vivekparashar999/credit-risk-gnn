"""
Training loops for the graph neural network, covering both modes the paper
implements (Section 4.5):

  Transductive -- the model sees the *whole* graph and *all* node features
      (train and test) during training, but is only supervised on the training
      labels. Test firms influence training indirectly, through message
      passing. This is the more accurate but less general setting.

  Inductive -- the model is trained on the training firms *only*: we hide the
      test nodes and any edge touching them. At prediction time the test firms
      (and their edges back into the trained graph) are revealed and pushed
      through the frozen network. This is how the model would actually be used
      on a brand-new firm that files with the SEC tomorrow.

Both modes share the same network (models.CreditGNN); only the graph the model
is allowed to look at changes. That is the whole point the paper makes -- the
architecture is identical, the *information exposure* is what differs.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from .models import CreditGNN


def _standardize(x_raw, train_idx):
    """Standardise features using training-firm statistics only. In the
    inductive setting it would be cheating to peek at test-firm features even
    for scaling, so we always fit the scaler on the train split."""
    scaler = StandardScaler().fit(x_raw[train_idx])
    return torch.as_tensor(scaler.transform(x_raw), dtype=torch.float32)


def _class_weights(y_train):
    """Inverse-frequency class weights. The data is imbalanced (~74% investment
    grade), and the paper optimises for F1 / macro-recall, so we tell the loss
    to care more about the rarer below-investment-grade class."""
    counts = np.bincount(y_train, minlength=2).astype(float)
    w = counts.sum() / (2.0 * np.clip(counts, 1, None))
    return torch.as_tensor(w, dtype=torch.float32)


def _restrict_edges_to(src, dst, keep_mask):
    """Keep only edges whose *both* endpoints are in the allowed node set --
    used to carve the training subgraph out of the full graph for inductive
    training."""
    ok = keep_mask[src] & keep_mask[dst]
    return src[ok], dst[ok]


def train_gnn(x_raw, src, dst, train_idx, test_idx, y, cfg, seed, mode):
    """Train the GNN in 'transductive' or 'inductive' mode and return the
    predicted probability of investment grade for the test firms."""
    torch.manual_seed(seed)
    n = x_raw.shape[0]

    x = _standardize(x_raw, train_idx)
    src_t = torch.as_tensor(src, dtype=torch.long)
    dst_t = torch.as_tensor(dst, dtype=torch.long)
    y_t = torch.as_tensor(y, dtype=torch.long)
    train_t = torch.as_tensor(train_idx, dtype=torch.long)

    # In inductive mode the model trains on the training subgraph only.
    if mode == "inductive":
        keep = np.zeros(n, dtype=bool)
        keep[train_idx] = True
        keep_t = torch.as_tensor(keep)
        train_src, train_dst = _restrict_edges_to(src_t, dst_t, keep_t)
    else:  # transductive: train on the full graph
        train_src, train_dst = src_t, dst_t

    model = CreditGNN(x.shape[1], cfg.hidden_dim, cfg.embed_dim, cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    weights = _class_weights(y[train_idx])

    model.train()
    for _ in range(cfg.epochs):
        opt.zero_grad()
        logits = model(x, train_src, train_dst)
        loss = F.cross_entropy(logits[train_t], y_t[train_t], weight=weights)
        loss.backward()
        opt.step()

    # Prediction: both modes use the full graph at inference time. For the
    # inductive model this is the first time it sees the test firms and their
    # edges -- exactly the "new firm arrives" scenario.
    model.eval()
    with torch.no_grad():
        logits = model(x, src_t, dst_t)
        prob = F.softmax(logits, dim=1)[:, 1].numpy()
    return prob[test_idx]
