"""
The graph neural network, written from scratch in plain PyTorch.

The paper uses GraphSAGE (Hamilton et al. 2017) with the max-pooling
aggregator -- "two GCN layers", "embedding of size 32", "no dropout",
trained end-to-end with the classifier. We implement exactly that here rather
than pulling in a heavy graph library, so every line of the message passing is
visible and runnable on a CPU.

One GraphSAGE layer does two things (Equations 2 and 3 in the paper):

  (2)  aggregate the neighbours:   h_N(i) = max_{j in N(i)} ReLU(W_pool h_j + b)
  (3)  combine with the node:      h_i'   = sigma( W [ h_i ; h_N(i) ] )

The "max" is element-wise max-pooling over the neighbour set. We compute it for
the whole graph at once using a scatter-reduce over the edge list, which is the
vectorised equivalent of looping over every node's neighbours.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGELayer(nn.Module):
    """One GraphSAGE layer with a max-pool aggregator."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pool = nn.Linear(in_dim, out_dim)            # W_pool, b  (Eq. 2)
        self.combine = nn.Linear(in_dim + out_dim, out_dim)  # W       (Eq. 3)

    def forward(self, h, src, dst):
        # Eq. 2: transform every node, ReLU, then max-pool over neighbours.
        msg = F.relu(self.pool(h))                        # [n, out]

        n, out = h.size(0), msg.size(1)
        # For a target node `dst`, its neighbours are the corresponding `src`.
        # scatter_reduce with amax = element-wise max over each node's inbox.
        agg = torch.zeros(n, out, device=h.device)
        index = dst.unsqueeze(1).expand(-1, out)
        agg = agg.scatter_reduce(0, index, msg[src], reduce="amax",
                                 include_self=False)
        # Isolated nodes receive no messages; scatter leaves them at 0, which
        # is the sensible "no neighbour information" default.

        # Eq. 3: concatenate the node's own vector with the aggregate, combine.
        return self.combine(torch.cat([h, agg], dim=1))


class CreditGNN(nn.Module):
    """Two SAGE layers -> node embedding -> a one-layer classifier head.

    The whole thing is trained end-to-end: the graph layers and the classifier
    learn their weights together against the classification loss, exactly as
    described in Section 4.4.
    """

    def __init__(self, in_dim, hidden_dim, embed_dim, dropout=0.0):
        super().__init__()
        self.sage1 = SAGELayer(in_dim, hidden_dim)
        self.sage2 = SAGELayer(hidden_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)
        self.dropout = dropout

    def forward(self, x, src, dst, return_embedding=False):
        h = F.relu(self.sage1(x, src, dst))
        if self.dropout:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.sage2(h, src, dst))               # node embedding (32-d)
        logits = self.classifier(h)
        if return_embedding:
            return logits, h
        return logits
