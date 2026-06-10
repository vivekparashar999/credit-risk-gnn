"""
Central place for every knob in the project.

Keeping the settings here (instead of scattering magic numbers across files)
means an experiment can be reproduced exactly by reading one file, and the
random seed makes every run repeatable.

The default values follow the choices reported in Das et al. (2023),
"Credit Risk Modeling with Graph Machine Learning":
  - embedding / hidden size 32        (Section 4.6)
  - two GCN layers                     (Section 4.4)
  - learning rate 0.02, weight decay 0.000294, ~120 epochs (Section 4.6, 5)
  - cosine-similarity cutoff 0.5 for building the graph (Section 4.1)
"""

from dataclasses import dataclass


@dataclass
class Config:
    # --- reproducibility ---
    seed: int = 42

    # --- synthetic data (paper, Appendix A) ---
    n_firms: int = 3000          # paper builds "more than 3,000 firms"
    noise: float = 0.30          # x = 0.3 in Table A.1 simulation equations
    text_dim: int = 64           # length of the per-firm MD&A embedding we simulate
    financial_corr: float = 0.72  # how strongly the financials track the rating
                                  # (<1 so the tabular model is good but beatable,
                                  #  matching the paper's ~0.73 tabular F1)

    # --- corporate graph (CorpNet) ---
    cosine_cutoff: float = 0.50  # link two firms if cosine similarity > cutoff

    # --- train / test split ---
    test_size: float = 0.30

    # --- GraphSAGE / GCN ---
    hidden_dim: int = 32
    embed_dim: int = 32
    lr: float = 0.02
    weight_decay: float = 0.000294
    epochs: int = 120
    dropout: float = 0.0         # paper reports "no dropout is applied"

    # how many times to repeat the experiment (averaged in the final tables)
    n_replications: int = 3


CFG = Config()
