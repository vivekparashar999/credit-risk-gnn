"""
Reproduce the synthetic-data experiment of Das et al. (2023),
"Credit Risk Modeling with Graph Machine Learning".

What this script does, end to end:
    1. generate the synthetic firms + their MD&A embeddings   (synthetic_data)
    2. build CorpNet from embedding similarity                (corpnet)
    3. add the three graph metrics to the tabular features    (corpnet)
    4. split into train / test
    5. fit and score three kinds of model:
         - tabular ML ensemble (the AutoGluon stand-in) -- the baseline
         - GraphSAGE GNN, transductive and inductive
         - the GNN + tabular probability ensemble
    6. repeat over `n_replications` runs and print the result tables, laid out
       like Table 1 in the paper (Panel A = tabular features only,
       Panel B = tabular features + three graph metrics).

Run:  python main.py
"""

import warnings
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import CFG
from src import synthetic_data, corpnet
from src.train import train_gnn
from src.tabular_ensemble import run_tabular
from src.metrics import evaluate, aggregate

warnings.filterwarnings("ignore")          # silence sklearn convergence chatter

TABULAR_COLS = ["A", "B", "C", "D", "E"]   # the five Altman ratios


def one_replication(seed, verbose=True):
    """Run the full pipeline once and return a {panel: {model: metrics}} dict."""
    say = print if verbose else (lambda *a, **k: None)

    # 1-3. data, graph, graph features
    df, emb = synthetic_data.generate(CFG.n_firms, CFG.noise, CFG.text_dim, seed,
                                      CFG.financial_corr)
    src, dst = corpnet.build_corpnet(emb, CFG.cosine_cutoff)
    g = corpnet.to_networkx(len(df), src, dst)
    info = corpnet.describe(g)
    say(f"   graph: {info['nodes']} nodes, {info['edges']} edges, "
        f"mean degree {info['mean_degree']:.1f}, {info['isolated']} isolated")

    graph_feats = corpnet.graph_node_features(g)
    y = df["label"].to_numpy()

    X_tab = df[TABULAR_COLS].to_numpy(dtype=np.float32)
    X_tab_graph = np.hstack([X_tab, graph_feats]).astype(np.float32)

    # 4. split (stratified so both classes appear in train and test)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=CFG.test_size, stratify=y, random_state=seed)
    y_train, y_test = y[train_idx], y[test_idx]

    # 5a. tabular ML -- once on plain features (Panel A), once with graph
    #     metrics added (Panel B)
    say("   fitting tabular ensemble (Panel A and B)...")
    tab_A = run_tabular(X_tab[train_idx], y_train, X_tab[test_idx], seed)
    tab_B = run_tabular(X_tab_graph[train_idx], y_train,
                        X_tab_graph[test_idx], seed)

    # 5b. GraphSAGE GNN -- node features are the five Altman ratios; the graph
    #     supplies the rest. Same model, two information regimes.
    say("   training GNN (transductive)...")
    gnn_t = train_gnn(X_tab, src, dst, train_idx, test_idx, y, CFG, seed,
                      mode="transductive")
    say("   training GNN (inductive)...")
    gnn_i = train_gnn(X_tab, src, dst, train_idx, test_idx, y, CFG, seed,
                      mode="inductive")

    # 5c. ensemble the GNN with the tabular ensemble by averaging probabilities
    ens = 0.5 * (gnn_t + tab_A["Ensemble (Tabular)"])

    def panel(tab_probs):
        return {
            "WeightedEnsemble (Tabular)": evaluate(y_test, tab_probs["Ensemble (Tabular)"]),
            "GNN (Transductive)": evaluate(y_test, gnn_t),
            "GNN (Inductive)": evaluate(y_test, gnn_i),
            "Ensemble: GNN+Tabular": evaluate(y_test, ens),
        }

    return {"Panel A (tabular only)": panel(tab_A),
            "Panel B (tabular + graph metrics)": panel(tab_B)}


def main():
    print(f"Running {CFG.n_replications} replication(s) of the synthetic-data "
          f"experiment (seed base = {CFG.seed})\n")

    # collect metrics per panel and model across replications
    collected = {}
    for rep in range(CFG.n_replications):
        print(f"[replication {rep + 1}/{CFG.n_replications}]")
        result = one_replication(CFG.seed + rep)
        for panel_name, models in result.items():
            for model_name, metrics in models.items():
                collected.setdefault(panel_name, {}).setdefault(
                    model_name, []).append(metrics)
        print()

    _print_tables(collected)


def _print_tables(collected):
    cols = ["F1", "Accuracy", "ROC_AUC", "MCC", "Mean recall", "Precision", "Recall"]
    for panel_name, models in collected.items():
        print("=" * 100)
        print(panel_name)
        print("=" * 100)
        header = f"{'Model':<28}" + "".join(f"{c:>11}" for c in cols)
        print(header)
        print("-" * len(header))
        for model_name, rows in models.items():
            mean, std = aggregate(rows)
            line = f"{model_name:<28}" + "".join(f"{mean[c]:>11.3f}" for c in cols)
            print(line)
            if len(rows) > 1:    # show the std line only when we replicated
                sub = f"{'':<28}" + "".join(f"{'('+format(std[c], '.3f')+')':>11}" for c in cols)
                print(sub)
        print()


if __name__ == "__main__":
    main()
