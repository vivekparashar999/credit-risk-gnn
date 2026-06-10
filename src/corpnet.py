"""
CorpNet -- the corporate graph -- and the three graph-derived node features.

Section 4.1 of the paper: take the document embeddings of every firm's MD&A
text, compute pairwise cosine similarity, and draw an undirected, unweighted
edge between two firms whenever their similarity exceeds a cutoff (0.5 by
default). The result is a homogeneous graph where "linked" means "the two
firms talk about their business in similar terms", which the paper argues is a
proxy for shared exposure to credit-risk spillovers.

Section 4.2 also describes three node-level metrics that summarise each firm's
position in the graph. Adding these three numbers to the tabular features is
the cheap, "old-school" way to let a non-graph model benefit from the network
(this is the paper's Panel B / Table A.3 setting).
"""

import numpy as np
import networkx as nx


def build_corpnet(embeddings, cutoff):
    """Build CorpNet from the firm embedding matrix.

    Returns the edge list as two parallel arrays (src, dst) containing *both*
    directions of every edge -- that symmetric form is what the message-passing
    code in models.py expects, and it matches an undirected graph.
    """
    # L2-normalize so that a dot product is exactly the cosine similarity.
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.clip(norm, 1e-12, None)
    sim = unit @ unit.T

    # We only want i<j pairs above the cutoff, and never a self-loop.
    np.fill_diagonal(sim, -1.0)
    iu, ju = np.where(np.triu(sim, k=1) > cutoff)

    # store both directions (i->j and j->i) for undirected message passing
    src = np.concatenate([iu, ju])
    dst = np.concatenate([ju, iu])
    return src.astype(np.int64), dst.astype(np.int64)


def to_networkx(n_nodes, src, dst):
    """A NetworkX view of the graph -- used for the centrality metrics and for
    node2vec's random walks. We add the i<j half only, since Graph() is
    undirected and would otherwise double-count."""
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    mask = src < dst
    g.add_edges_from(zip(src[mask].tolist(), dst[mask].tolist()))
    return g


def graph_node_features(g):
    """The three metrics the paper adds to the tabular data (Section 4.2):

      - degree centrality:     how many firms a firm is linked to (normalised);
      - eigenvector centrality: connectedness weighted by how important the
                                 neighbours themselves are;
      - clustering coefficient: how tightly a firm's neighbours link to each
                                 other (do its peers form a clique?).

    All three are standard interconnectedness measures from the systemic-risk
    literature, which is exactly why the authors picked them.
    """
    n = g.number_of_nodes()
    degree = nx.degree_centrality(g)
    clustering = nx.clustering(g)
    try:
        eigen = nx.eigenvector_centrality(g, max_iter=1000, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        # falls back gracefully on rare non-converging graphs
        eigen = {i: 0.0 for i in range(n)}

    feats = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        feats[i] = (degree.get(i, 0.0), eigen.get(i, 0.0), clustering.get(i, 0.0))
    return feats


def describe(g):
    """A few summary numbers so we can sanity-check the graph against the
    paper (it reports a mean degree of ~34-110 and a scale-free shape)."""
    degrees = [d for _, d in g.degree()]
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "isolated": int(sum(d == 0 for d in degrees)),
    }
