"""
The tabular machine-learning baseline.

The paper uses AutoGluon, an AutoML library that trains a zoo of models
(random forests, extra-trees, LightGBM/CatBoost/XGBoost, k-NN, neural nets)
and then stacks them into a 'WeightedEnsemble_L2'. AutoGluon is a heavy
dependency and a black box, so to keep this project transparent and laptop-
friendly we rebuild the *idea* by hand: train a handful of strong, diverse
models and average their predicted probabilities into one ensemble.

This is faithful to what matters for the paper's argument -- the tabular model
is meant to be a strong, hard-to-beat benchmark, and a soft-voting ensemble of
forests + gradient boosting is exactly that. We report both the individual
models and the ensemble, mirroring Tables A.2/A.3.
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def _base_models(seed):
    """The diverse set of base learners we ensemble. Each one captures a
    different kind of structure, which is why averaging them helps."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=seed, n_jobs=-1),
        "LogReg": LogisticRegression(max_iter=1000),
    }


def run_tabular(X_train, y_train, X_test, seed):
    """Fit every base model and the soft-voting ensemble.

    Returns a dict {model_name: predicted_probability_of_class_1 on X_test},
    including an "Ensemble (Tabular)" entry. The features are standardised
    using statistics from the *training* split only, so no test information
    leaks into the fit (this matters for a fair inductive comparison).
    """
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    probs = {}
    for name, model in _base_models(seed).items():
        model.fit(Xtr, y_train)
        probs[name] = model.predict_proba(Xte)[:, 1]

    # AutoGluon's WeightedEnsemble_L2, simplified to an equal-weight average.
    probs["Ensemble (Tabular)"] = np.mean(list(probs.values()), axis=0)
    return probs
