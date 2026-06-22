"""
Core data + model helpers for Project 2 (Explainability).

We work with the Palmer Penguins dataset. The target is ``species`` (three
classes). We train two families of models:

* a Decision Tree family, regularised via ``max_leaf_nodes``;
* a Logistic Regression family, regularised via an L1 penalty (strength ``C``).

For each family we expose, per fitted model, its test accuracy and a complexity
measure ``Omega``:

* tree   -> number of leaves
* logreg -> number of non-zero coefficients (L1 sparsity)

Task 2/3 then pick, for a given slider value ``lambda``, the member of the
family that maximises ``acc_test - lambda * Omega``.
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # file backend; no GUI on a web server
import matplotlib.pyplot as plt

from django.conf import settings

from palmerpenguins import load_penguins

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Where generated images live.
DATA_DIR = os.path.join(settings.MEDIA_ROOT, "project2")

# Feature groups. The target is ``species``.
TARGET = "species"
NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex", "year"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Fixed split so every request sees the same train/test partition.
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Hyperparameter grids for the two families.
TREE_LEAF_GRID = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30]
# Inverse regularisation strength for logistic regression (small C -> sparser).
LOGREG_C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_data():
    """Load and clean the penguins dataset, returning a train/test split.

    Returns a dict with the raw cleaned frame plus X/y train/test partitions
    (as DataFrames/Series so column names survive into the pipelines).
    """
    df = load_penguins()
    # ``year`` is read as an int; keep categoricals as strings for the encoders.
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return {
        "df": df,
        "classes": sorted(y.unique().tolist()),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ----------------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------------
def _tree_pipeline(max_leaf_nodes):
    """Tree: ordinal-encode categoricals, pass numerics through (no scaling)."""
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             CATEGORICAL_FEATURES),
        ]
    )
    clf = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes, random_state=RANDOM_STATE
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _logreg_pipeline(C):
    """LogReg: one-hot categoricals, standardise numerics, L1 penalty."""
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    # saga + multinomial gives a true softmax model, so ALE can use the exact
    # analytic gradient of the class probabilities (Task 5).
    clf = LogisticRegression(
        penalty="l1", solver="saga", multi_class="multinomial",
        C=C, max_iter=10000, tol=1e-3, random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


# ----------------------------------------------------------------------------
# Complexity measures Omega
# ----------------------------------------------------------------------------
def tree_omega(pipe):
    """Omega(tree) = number of leaves."""
    return int(pipe.named_steps["clf"].get_n_leaves())


def logreg_omega(pipe):
    """Omega(logreg) = number of non-zero coefficients (L1 sparsity)."""
    coef = pipe.named_steps["clf"].coef_
    return int(np.count_nonzero(coef))


# ----------------------------------------------------------------------------
# Train a whole family
# ----------------------------------------------------------------------------
@lru_cache(maxsize=2)
def train_family(model_class):
    """Fit every member of a family; return a list of result dicts (cached).

    Each entry: {param, pipe, acc_test, acc_train, omega}.
    """
    data = get_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    if model_class == "tree":
        grid, build, omega_fn, param_name = (
            TREE_LEAF_GRID, _tree_pipeline, tree_omega, "max_leaf_nodes")
    elif model_class == "logreg":
        grid, build, omega_fn, param_name = (
            LOGREG_C_GRID, _logreg_pipeline, logreg_omega, "C")
    else:
        raise ValueError(f"unknown model class: {model_class}")

    results = []
    for param in grid:
        pipe = build(param)
        pipe.fit(X_train, y_train)
        results.append({
            "param": param,
            "param_name": param_name,
            "pipe": pipe,
            "acc_train": round(float(accuracy_score(y_train, pipe.predict(X_train))), 4),
            "acc_test": round(float(accuracy_score(y_test, pipe.predict(X_test))), 4),
            "omega": omega_fn(pipe),
        })
    return results


def select_model(model_class, lam):
    """Pick the family member maximising ``acc_test - lambda * Omega`` (Task 2/3).

    Returns (best_entry, full_family). Ties are broken towards the simpler model
    (smaller Omega), since the family is iterated from simple to complex.
    """
    family = train_family(model_class)
    best, best_score = None, None
    for entry in family:
        score = entry["acc_test"] - lam * entry["omega"]
        entry["score"] = round(score, 4)
        if best_score is None or score > best_score:
            best, best_score = entry, score
    return best, family


# ----------------------------------------------------------------------------
# Model visualisation (Task 1 / Task 3)
# ----------------------------------------------------------------------------
def logreg_feature_names(pipe):
    """Feature names produced by the logistic-regression preprocessor."""
    return list(pipe.named_steps["pre"].get_feature_names_out())


def render_model(pipe, model_class):
    """Draw the selected model to media/ and return the image URL.

    * tree   -> the actual decision tree (plot_tree)
    * logreg -> a heatmap of the per-class coefficients
    """
    _ensure_dir()
    classes = get_data()["classes"]

    if model_class == "tree":
        clf = pipe.named_steps["clf"]
        fig, ax = plt.subplots(figsize=(min(2.2 * clf.get_n_leaves(), 22), 9))
        plot_tree(
            clf, feature_names=FEATURES, class_names=classes,
            filled=True, rounded=True, impurity=False, fontsize=9, ax=ax,
        )
        fname = "tree.png"
    else:
        clf = pipe.named_steps["clf"]
        names = logreg_feature_names(pipe)
        coef = clf.coef_                       # shape (n_classes, n_features)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(names))))
        vmax = np.abs(coef).max() or 1.0
        im = ax.imshow(coef.T, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(clf.classes_)))
        ax.set_xticklabels(clf.classes_)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_title("Logistic-regression coefficients (zero = pruned by L1)")
        for i in range(coef.shape[1]):          # annotate each cell
            for j in range(coef.shape[0]):
                ax.text(j, i, f"{coef[j, i]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(coef[j, i]) > 0.6 * vmax else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fname = "logreg_coef.png"

    fig.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, fname), dpi=110)
    plt.close(fig)
    return settings.MEDIA_URL + "project2/" + fname
