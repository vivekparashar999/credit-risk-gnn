"""
Synthetic data generation -- a direct implementation of Appendix A of
Das et al. (2023).

Why synthetic and not the "real" Kaggle data?
    The paper uses two datasets. The real one needs (a) a Kaggle credit-rating
    file and (b) the actual text of thousands of SEC 10-K/10-Q filings pulled
    through AWS SageMaker JumpStart. Reproducing that needs paid AWS tooling
    and is not something we can run on a laptop. The authors anticipated this,
    so they *released a synthetic generator* (Appendix A) precisely so that
    others can reproduce the whole pipeline. That is what we implement here.

The generator has three jobs:
    1. Simulate each firm's balance-sheet / income-statement items, calibrated
       to U.S.-economy averages (Table A.1), and compute Altman's Z-score.
    2. Produce a per-firm "MD&A text" signal. In the paper this comes from
       doc2vec embeddings of real filings; we simulate an embedding vector that
       carries a shared 'credit-quality' signal plus topic structure, because
       that is exactly what makes the corporate graph informative.
    3. Assign credit ratings by ranking firms and slicing them into the rating
       proportions the paper quotes (AAA 3% ... CCC+ 1%), then collapse to the
       binary investment-grade / below-investment-grade label.

The single most important modelling fact, straight from the paper: the ratings
and the graph both originate from the same MD&A text. That shared origin is why
two firms that are *linked* in the graph tend to share a rating -- and that is
the signal a graph neural network can exploit but a plain tabular model cannot.
"""

import numpy as np
import pandas as pd

# Table A.1 -- normalized average values for U.S. companies (2021 Q1),
# expressed as a percentage of total assets (which is fixed at 100).
BASELINE = {
    "TA": 100.00,   # total assets (held fixed; everything is normalized by it)
    "CL": 19.37,    # current liabilities
    "TL": 58.67,    # total liabilities
    "EQ": 12.75,    # book equity
    "RE": 28.58,    # retained earnings
    "CA": 26.54,    # current assets
    "NS": 13.99,    # net sales
    "EBIT": 1.32,   # earnings before interest and taxes
    "P2B": 4.75,    # price-to-book ratio (a ratio, not a % of assets)
}

# Rating buckets and their share of the population (Appendix A.1).
# Investment grade = AAA..BBB; below investment grade = BB, B, CCC+.
RATING_BUCKETS = [
    ("AAA", 0.03), ("AA", 0.17), ("A", 0.30), ("BBB", 0.24),
    ("BB", 0.13), ("B", 0.12), ("CCC+", 0.01),
]
INVESTMENT_GRADE = {"AAA", "AA", "A", "BBB"}


def _simulate_one_firm(rng, noise):
    """Simulate the eight Altman inputs for a single firm and return them,
    or return None if the firm fails the balance-sheet consistency checks.

    Each item is the baseline scaled by (1 + x*(u - 0.5)) with u ~ U(0,1),
    i.e. a multiplicative jitter of +/- x/2 around the average. This is the
    'Simulation equation' column of Table A.1.
    """
    def jitter(value):
        u = rng.random()
        return value * (1 + noise * (u - 0.5))

    TA = BASELINE["TA"]              # fixed at 100
    CL = jitter(BASELINE["CL"])
    TL = jitter(BASELINE["TL"])
    EQ = jitter(BASELINE["EQ"])
    RE = jitter(BASELINE["RE"])
    CA = jitter(BASELINE["CA"])
    NS = jitter(BASELINE["NS"])
    EBIT = jitter(BASELINE["EBIT"])
    P2B = jitter(BASELINE["P2B"])

    MVE = P2B * (EQ + RE)            # market value of equity
    WC = CA - CL                     # working capital

    # Consistency checks from Table A.1 -- discard firms whose simulated books
    # do not hold together, so we never train on nonsensical balance sheets.
    if not (TL > CL and TA >= EQ + RE + TL and TA > CA and NS > EBIT and P2B > 0):
        return None

    return {"TA": TA, "CL": CL, "TL": TL, "EQ": EQ, "RE": RE,
            "CA": CA, "NS": NS, "EBIT": EBIT, "P2B": P2B,
            "MVE": MVE, "WC": WC}


def _altman_ratios(f):
    """The five ratios of the Altman (1968) Z-score model and the Z-score
    itself (Equation A.1). These five ratios are the tabular features."""
    A = f["EBIT"] / f["TA"]          # profitability
    B = f["NS"] / f["TA"]            # asset turnover
    C = f["MVE"] / f["TL"]           # market leverage
    D = f["WC"] / f["TA"]            # short-term liquidity
    E = f["RE"] / f["TA"]            # cumulative profitability
    z = 3.3 * A + 0.99 * B + 0.6 * C + 1.2 * D + 1.4 * E
    return A, B, C, D, E, z


def _simulate_text_embeddings(rng, n, dim):
    """Stand in for doc2vec embeddings of the MD&A sections.

    Real doc2vec vectors of corporate filings are not spread evenly over the
    sphere: every firm writes about "business", "risk", "markets", so the
    vectors sit inside a narrow cone and most pairwise cosine similarities are
    positive. That is exactly why a 0.5 cutoff keeps only a couple of percent
    of pairs yet still produces a well-connected graph. We reproduce that with
    four parts:
      - a shared positive base vector (the "everyone talks about business"
        component) that pushes cosine similarities into the positive range and
        controls how dense the graph comes out;
      - a 'credit-quality' axis (q): the same latent that drives the firm's
        rating, so text similarity lines up with rating similarity (homophily);
      - a few sparse, heavy-tailed 'topic' axes: these create clusters and a
        few highly-connected hub firms -> the scale-free degree distribution
        the paper reports;
      - small isotropic noise.

    Returns the embedding matrix and the quality scores q (used for ranking).
    """
    # quality latent -- this is what ratings will ultimately be ranked on
    q = rng.standard_normal(n)

    base = 0.16 * (np.abs(rng.standard_normal(dim)) + 0.5)  # shared, all-positive

    quality_axis = rng.standard_normal(dim)
    quality_axis /= np.linalg.norm(quality_axis)

    n_topics = 8
    topic_axes = rng.standard_normal((n_topics, dim))
    topic_axes /= np.linalg.norm(topic_axes, axis=1, keepdims=True)
    # heavy-tailed, half-sparse topic loadings -> clusters + a few hub firms
    loadings = rng.exponential(scale=1.0, size=(n, n_topics))
    loadings *= (rng.random((n, n_topics)) < 0.5)   # zero out ~50% of loadings

    quality_strength = 1.0   # how strongly text reflects credit quality
    emb = (
        base[None, :]
        + quality_strength * q[:, None] * quality_axis[None, :]
        + 1.0 * (loadings @ topic_axes)
        + 0.4 * rng.standard_normal((n, dim))
    )
    return emb, q


def generate(n_firms, noise, text_dim, seed, financial_corr=0.72):
    """Build the full synthetic dataset.

    Returns
    -------
    df  : DataFrame with the five Altman ratios, the Z-score, the rating, and
          the binary label (1 = investment grade, 0 = below).
    emb : (n, text_dim) matrix of simulated MD&A embeddings, aligned row-for-row
          with df -- this is the input to the graph builder.
    """
    rng = np.random.default_rng(seed)

    # --- Step 3: simulate financials until we have n_firms valid ones ---
    firms = []
    while len(firms) < n_firms:
        f = _simulate_one_firm(rng, noise)
        if f is not None:
            firms.append(f)

    ratios = [_altman_ratios(f) for f in firms]
    fin = pd.DataFrame(ratios, columns=["A", "B", "C", "D", "E", "Zscore"])

    # --- Steps 1-2: simulate text and its quality signal ---
    emb, q = _simulate_text_embeddings(rng, n_firms, text_dim)

    # --- Step 4: align and label (the paper's "sort each side, concatenate") ---
    # The rating is driven by the TEXT quality (the paper assigns ratings from
    # the text net-score), so we order firms by q and keep each firm's
    # embedding with its rating. This is what makes the graph homophilous.
    txt_order = np.argsort(-q)                  # best text quality first
    emb_sorted = emb[txt_order]
    ratings = _assign_ratings_by_rank(n_firms)  # rank position -> rating bucket

    # The financials should track the rating, but only loosely -- otherwise the
    # Z-score would predict the label perfectly and there would be nothing left
    # for the graph to add. We build a noisy proxy `fq` of the (now ranked)
    # quality, then hand each firm the financials whose Z-score rank matches its
    # fq rank. `financial_corr` controls how tight that link is.
    q_sorted = q[txt_order]
    q_std = (q_sorted - q_sorted.mean()) / q_sorted.std()
    fq = (financial_corr * q_std
          + np.sqrt(1 - financial_corr ** 2) * rng.standard_normal(n_firms))

    fin_by_z = np.argsort(-fin["Zscore"].to_numpy())   # financial rows, best Z first
    firm_by_fq = np.argsort(-fq)                        # firm positions, best fq first
    assigned = np.empty(n_firms, dtype=int)
    assigned[firm_by_fq] = fin_by_z                    # firm position -> financial row
    fin_aligned = fin.iloc[assigned].reset_index(drop=True)

    df = fin_aligned.copy()
    df["rating"] = ratings
    df["label"] = df["rating"].isin(INVESTMENT_GRADE).astype(int)
    return df, emb_sorted


def _assign_ratings_by_rank(n):
    """Slice n rank-ordered firms (best first) into the rating buckets using
    the paper's quoted proportions."""
    ratings = np.empty(n, dtype=object)
    start = 0
    for name, share in RATING_BUCKETS:
        end = start + int(round(share * n))
        ratings[start:end] = name
        start = end
    # rounding can leave the tail unfilled -- give any leftover firms the
    # lowest bucket
    if start < n:
        ratings[start:] = RATING_BUCKETS[-1][0]
    return ratings
