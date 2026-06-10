# Credit Risk Modeling with Graph Machine Learning — Code Implementation

A runnable implementation of the methodology in:

> **Das, Huang, Adeshina, Yang, Bachega (2023).** *Credit Risk Modeling with
> Graph Machine Learning.* INFORMS Journal on Data Science 2(2):197–217.

The paper's idea: traditional credit-rating models use only **tabular**
financial data (balance-sheet / income-statement ratios). This work augments
that with a **corporate graph (CorpNet)** built from the text of SEC filings,
and shows that a **graph neural network (GNN)** trained on the graph +
financials predicts investment-grade vs below-investment-grade ratings as well
as or better than a strong tabular model.

This repo reproduces the pipeline on the paper's **synthetic dataset**
(Appendix A), released by the authors so the method can be reproduced without
the paid AWS/SEC tooling the real dataset needs.

## What's in here

| File | Role | Paper section |
|------|------|---------------|
| `src/synthetic_data.py` | synthetic firms (Altman Z-score) + simulated MD&A embeddings + ratings | §3.2, App. A |
| `src/corpnet.py` | build CorpNet from embedding similarity (cutoff 0.5) + 3 graph metrics | §4.1–4.2 |
| `src/models.py`, `src/train.py` | GraphSAGE GNN (Eqs. 2–3), transductive & inductive | §4.4–4.6 |
| `src/tabular_ensemble.py` | tabular ML ensemble — the baseline | §4.3 |
| `src/metrics.py`, `main.py` | evaluation + result tables | §5 |

Two notes: the GraphSAGE layers are written directly in PyTorch (no DGL /
PyTorch-Geometric needed), and AutoGluon is replaced by a soft-voting ensemble
of standard sklearn/XGBoost models, which plays the same role as a strong
tabular benchmark without the heavy dependency.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

It prints two tables like the paper's Table 1 — **Panel A** (tabular features
only) and **Panel B** (tabular + the three graph node-metrics). Settings live in
`src/config.py`.

## Results

Run on 3,000 synthetic firms (full output in `results.txt`), **Panel A**:

| Model | F1 | Accuracy | ROC-AUC | MCC | Mean recall |
|-------|----|----------|---------|-----|-------------|
| Tabular ensemble | 0.883 | 0.820 | 0.850 | 0.498 | 0.724 |
| GNN (Transductive) | 0.891 | 0.846 | **0.928** | 0.637 | **0.840** |
| GNN (Inductive) | 0.890 | 0.845 | 0.924 | 0.635 | 0.837 |
| GNN + Tabular ensemble | **0.904** | **0.859** | 0.919 | **0.643** | 0.828 |

The GNN beats the tabular baseline on the metrics that matter for an imbalanced
problem (ROC-AUC, MCC, macro-recall), and the GNN+tabular ensemble gives the
best F1. In Panel B, adding the three graph metrics also lifts the plain tabular
model — so the corporate graph carries real credit-risk signal beyond the
financial ratios alone, which is the paper's conclusion.

Numbers vary slightly per run because the data is generated randomly. Absolute
values on synthetic data are not meant to match the paper's real-data tables;
the ordering between models is what reproduces, and it does.
