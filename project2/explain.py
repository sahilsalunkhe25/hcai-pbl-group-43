"""
Explainability helpers for Project 2 (Palmer Penguins).

All data loading, model fitting, counterfactual generation and the
from-scratch PDP/ALE computations live here so the view stays a thin
request/response layer, mirroring project1/ml.py.
"""

import os
import warnings
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")          # no GUI on a web server -> use the file backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from django.conf import settings

from palmerpenguins import load_penguins
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
TARGET = "species"
NUMERIC = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL = ["island", "sex", "year"]
FEATURES = NUMERIC + CATEGORICAL

DATA_DIR = os.path.join(settings.MEDIA_ROOT, "project2")

MODEL_LABELS = {"tree": "Decision Tree", "logreg": "Logistic Regression (L1)"}
COMPLEXITY_LABELS = {"tree": "leaves", "logreg": "nonzero coefficients"}

# max_leaf_nodes values tried for the tree; None = fully grown / unconstrained
TREE_LEAF_GRID = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, None]
LOGREG_C_GRID = np.logspace(-3, 2, 15)

LAMBDA_MAX = 0.05
LAMBDA_STEP = 0.001

PDP_POINTS = 25
ALE_BINS = 15


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


@lru_cache(maxsize=1)
def _dataset():
    df = load_penguins().dropna().reset_index(drop=True)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )
    return {
        "df": df,
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }


def get_species_list():
    return sorted(_dataset()["df"][TARGET].unique().tolist())


@lru_cache(maxsize=1)
def _mad():
    """Median absolute deviation per numeric feature (train set), floored to avoid /0."""
    X_train = _dataset()["X_train"]
    out = {}
    for f in NUMERIC:
        vals = X_train[f].to_numpy(dtype=float)
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        out[f] = float(mad) if mad > 1e-6 else 1e-6
    return out


@lru_cache(maxsize=1)
def _category_values():
    X_train = _dataset()["X_train"]
    return {f: sorted(X_train[f].unique().tolist(), key=str) for f in CATEGORICAL}


# ----------------------------------------------------------------------------
# Shared preprocessing: numeric columns come first in the ColumnTransformer's
# output, in NUMERIC order, so transformed column i == NUMERIC[i]. This is
# relied on later to read out the exact logistic-regression derivative.
# ----------------------------------------------------------------------------
def _make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        sparse_threshold=0,
    )


def _transformed_feature_names(pipeline):
    pre = pipeline.named_steps["pre"]
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL)
    return NUMERIC + list(cat_names)


# ----------------------------------------------------------------------------
# Tasks 1/2 — Decision tree grid (Ω = number of leaves)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _tree_grid():
    data = _dataset()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    grid = []
    for max_leaf_nodes in TREE_LEAF_GRID:
        clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=42)
        pipe = Pipeline([("pre", _make_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        grid.append({
            "param": max_leaf_nodes,
            "complexity": clf.get_n_leaves(),
            "test_acc": accuracy_score(y_test, pipe.predict(X_test)),
            "pipeline": pipe,
        })
    return grid


# ----------------------------------------------------------------------------
# Task 3 — Logistic regression grid (Ω = number of nonzero coefficients)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _logreg_grid():
    data = _dataset()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    grid = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for C in LOGREG_C_GRID:
            clf = LogisticRegression(
                C=C, penalty="l1", solver="saga", multi_class="multinomial",
                max_iter=5000, random_state=42,
            )
            pipe = Pipeline([("pre", _make_preprocessor()), ("clf", clf)])
            pipe.fit(X_train, y_train)
            grid.append({
                "param": float(C),
                "complexity": int(np.sum(np.abs(clf.coef_) > 1e-6)),
                "test_acc": accuracy_score(y_test, pipe.predict(X_test)),
                "pipeline": pipe,
            })
    return grid


def get_grid(model_class):
    return _tree_grid() if model_class == "tree" else _logreg_grid()


def select_best(grid, lam):
    """The maximizer of acc_test - lambda * Omega(f), eq. (1) in the handout."""
    return max(grid, key=lambda e: e["test_acc"] - lam * e["complexity"])


# ----------------------------------------------------------------------------
# Plotting: the fitted model (Tasks 1-3)
# ----------------------------------------------------------------------------
def plot_tree_model(entry):
    _ensure_dir()
    pipe = entry["pipeline"]
    clf = pipe.named_steps["clf"]
    feature_names = _transformed_feature_names(pipe)
    class_names = [str(c) for c in clf.classes_]

    fig_w = max(10, clf.get_n_leaves() * 1.1)
    plt.figure(figsize=(fig_w, 8))
    plot_tree(clf, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "tree.png"), dpi=110)
    plt.close()
    return settings.MEDIA_URL + "project2/tree.png"


def plot_logreg_coefs(entry):
    _ensure_dir()
    pipe = entry["pipeline"]
    clf = pipe.named_steps["clf"]
    feature_names = _transformed_feature_names(pipe)
    class_names = [str(c) for c in clf.classes_]
    coef = clf.coef_

    n_features = len(feature_names)
    x = np.arange(n_features)
    width = 0.8 / len(class_names)

    plt.figure(figsize=(max(10, n_features * 0.7), 6))
    for i, cname in enumerate(class_names):
        plt.bar(x + i * width, coef[i], width=width, label=cname)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x + width * (len(class_names) - 1) / 2, feature_names, rotation=45, ha="right")
    plt.ylabel("Coefficient (standardized numeric features)")
    plt.title("Logistic regression coefficients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "logreg.png"), dpi=110)
    plt.close()
    return settings.MEDIA_URL + "project2/logreg.png"


def plot_model(model_class, entry):
    return plot_tree_model(entry) if model_class == "tree" else plot_logreg_coefs(entry)


# ----------------------------------------------------------------------------
# Counterfactual examples (Task 4)
# ----------------------------------------------------------------------------
def get_example_choices(limit=30):
    data = _dataset()
    X_test, y_test = data["X_test"], data["y_test"]
    n = min(limit, len(X_test))
    choices = []
    for i in range(n):
        row = X_test.iloc[i]
        choices.append({
            "idx": i,
            "label": f"#{i} – {y_test.iloc[i]}, {row['island']}, {row['sex']}, "
                     f"{row['bill_length_mm']:.1f}mm bill",
        })
    return choices


def get_example_row(idx):
    X_test = _dataset()["X_test"]
    if idx is None or idx < 0 or idx >= len(X_test):
        return None
    return X_test.iloc[idx].to_dict()


def _fmt(feature, value):
    if feature in NUMERIC:
        return round(float(value), 1)
    return str(value)


def generate_counterfactuals(example, target_class, pipeline, k=5):
    """
    Randomly sample points locally around `example`, keep the ones the model
    classifies as `target_class`, and rank them by MAD-weighted L1 distance.

    Numeric features are perturbed with Gaussian noise scaled by their MAD;
    categorical features (island/sex/year) are, with some probability,
    resampled to a different value observed in the training data. If nothing
    is found, both the sample count and the noise scale are grown and the
    search is retried.
    """
    mad = _mad()
    cat_values = _category_values()
    rng = np.random.default_rng()
    classes = list(pipeline.named_steps["clf"].classes_)
    target_idx = classes.index(target_class)

    N, scale = 200, 1.0
    kept = None
    for _ in range(5):
        rows = []
        for _ in range(N):
            new = dict(example)
            for f in NUMERIC:
                new[f] = float(example[f]) + rng.normal(0, scale * mad[f])
            for f in CATEGORICAL:
                if rng.random() < min(0.3 * scale, 0.9):
                    choices = [v for v in cat_values[f] if v != example[f]]
                    if choices:
                        new[f] = choices[rng.integers(len(choices))]
            rows.append(new)

        cand_df = pd.DataFrame(rows)[FEATURES]
        preds = pipeline.predict(cand_df)
        proba = pipeline.predict_proba(cand_df)
        mask = preds == target_class
        if mask.any():
            kept = cand_df[mask].copy()
            kept["_target_proba"] = proba[mask, target_idx]
            break
        N *= 2
        scale *= 1.5

    if kept is None or kept.empty:
        return {"found": False, "features": FEATURES, "example_cells": [], "rows": []}

    def distance(row):
        d = sum(abs(float(row[f]) - float(example[f])) / mad[f] for f in NUMERIC)
        d += sum(0.0 if row[f] == example[f] else 1.0 for f in CATEGORICAL)
        return d

    kept["_distance"] = kept.apply(distance, axis=1)
    kept = kept.sort_values("_distance").head(k)

    example_fmt = {f: _fmt(f, example[f]) for f in FEATURES}
    example_cells = [{"feature": f, "value": example_fmt[f]} for f in FEATURES]

    out_rows = []
    for r in kept.to_dict("records"):
        values = {f: _fmt(f, r[f]) for f in FEATURES}
        cells = [
            {"feature": f, "value": values[f], "changed": values[f] != example_fmt[f]}
            for f in FEATURES
        ]
        out_rows.append({
            "cells": cells,
            "target_proba": round(float(r["_target_proba"]), 3),
            "distance": round(float(r["_distance"]), 2),
        })

    return {"found": True, "features": FEATURES, "example_cells": example_cells, "rows": out_rows}


# ----------------------------------------------------------------------------
# Task 5 — Partial Dependence and Accumulated Local Effects, written by hand.
# ----------------------------------------------------------------------------
def compute_pdp(pipeline, feature, n_points=PDP_POINTS):
    """
    PDP(v) = average over the training set of predict_proba with `feature`
    clamped to v for every row. No derivatives involved.
    """
    X_train = _dataset()["X_train"]
    classes = list(pipeline.named_steps["clf"].classes_)
    grid = np.linspace(X_train[feature].min(), X_train[feature].max(), n_points)

    values = np.zeros((n_points, len(classes)))
    for i, v in enumerate(grid):
        X_mod = X_train.copy()
        X_mod[feature] = v
        values[i] = pipeline.predict_proba(X_mod).mean(axis=0)
    return grid, values, classes


def compute_ale(pipeline, feature, model_kind, n_bins=ALE_BINS):
    """
    Accumulated Local Effects for `feature`.

    Local effect per quantile bin:
      - logistic regression: the model is linear in log-odds, so the exact
        partial derivative of the softmax probability w.r.t. the *original*
        feature is available in closed form:
            dp_c/dx_j = p_c * (beta_{j,c} - sum_c' p_c' * beta_{j,c'})
        (beta_{j,c} is the fitted coefficient undone from standardization).
        Evaluated at each row in the bin, averaged, times the bin width.
      - decision tree: predict_proba is piecewise constant, so the
        derivative is zero almost everywhere and undefined at the splits.
        No exact derivative exists, so the local effect is instead
        approximated (discretized) by the finite difference of predictions
        at the bin's upper vs. lower edge.

    The resulting step function is accumulated (cumulative sum across bins)
    and mean-centered, per the standard ALE definition.
    """
    X_train = _dataset()["X_train"]
    classes = list(pipeline.named_steps["clf"].classes_)
    n_classes = len(classes)
    values = X_train[feature].to_numpy(dtype=float)

    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    K = len(quantiles) - 1
    if K < 1:
        return quantiles, np.zeros((len(quantiles), n_classes)), classes

    bin_idx = np.clip(np.digitize(values, quantiles[1:-1], right=True), 0, K - 1)

    local = np.zeros((K, n_classes))
    counts = np.zeros(K)

    beta = None
    if model_kind == "logreg":
        clf = pipeline.named_steps["clf"]
        scaler = pipeline.named_steps["pre"].named_transformers_["num"]
        f_idx = NUMERIC.index(feature)
        beta = clf.coef_[:, f_idx] / scaler.scale_[f_idx]

    for k in range(K):
        rows = X_train[bin_idx == k]
        if rows.empty:
            continue
        lo, hi = quantiles[k], quantiles[k + 1]
        counts[k] = len(rows)

        if model_kind == "logreg":
            proba_rows = pipeline.predict_proba(rows)
            dot = proba_rows @ beta
            deriv = proba_rows * (beta[None, :] - dot[:, None])
            local[k] = deriv.mean(axis=0) * (hi - lo)
        else:
            rows_hi, rows_lo = rows.copy(), rows.copy()
            rows_hi[feature], rows_lo[feature] = hi, lo
            local[k] = (pipeline.predict_proba(rows_hi) - pipeline.predict_proba(rows_lo)).mean(axis=0)

    accumulated = np.vstack([np.zeros(n_classes), np.cumsum(local, axis=0)])
    midpoints = (accumulated[:-1] + accumulated[1:]) / 2
    total = counts.sum()
    weighted_mean = (midpoints * counts[:, None]).sum(axis=0) / total if total > 0 else 0.0
    centered = accumulated - weighted_mean

    return quantiles, centered, classes


def plot_pdp_ale(feature, pdp_grid, pdp_values, ale_edges, ale_values, classes):
    _ensure_dir()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, (ax_pdp, ax_ale) = plt.subplots(1, 2, figsize=(12, 5))
    for i, cls in enumerate(classes):
        color = colors[i % len(colors)]
        ax_pdp.plot(pdp_grid, pdp_values[:, i], label=str(cls), color=color)
        ax_ale.plot(ale_edges, ale_values[:, i], label=str(cls), color=color)

    ax_pdp.set_title(f"PDP – {feature}")
    ax_pdp.set_xlabel(feature)
    ax_pdp.set_ylabel("Predicted probability")
    ax_pdp.legend()
    ax_pdp.grid(alpha=0.3)

    ax_ale.axhline(0, color="black", linewidth=0.8)
    ax_ale.set_title(f"ALE – {feature}")
    ax_ale.set_xlabel(feature)
    ax_ale.set_ylabel("Accumulated local effect")
    ax_ale.legend()
    ax_ale.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "effects.png"), dpi=110)
    plt.close()
    return settings.MEDIA_URL + "project2/effects.png"
