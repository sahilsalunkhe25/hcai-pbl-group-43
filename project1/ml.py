"""
Machine-learning helpers for Project 1.

All the data handling, plotting and the training pipeline live here so the
views stay short and only deal with the web (requests / responses).
"""

import os

import matplotlib
matplotlib.use("Agg")          # no GUI on a web server -> use the file backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from django.conf import settings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score


# Where we keep the uploaded dataset and the generated images.
DATA_DIR = os.path.join(settings.MEDIA_ROOT, "project1")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")


# ----------------------------------------------------------------------------
# Dataset loading / saving
# ----------------------------------------------------------------------------
def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def save_uploaded_dataset(uploaded_file):
    """Persist the uploaded CSV so later views (explore/train) can re-read it."""
    _ensure_dir()
    with open(DATASET_PATH, "wb") as out:
        for chunk in uploaded_file.chunks():
            out.write(chunk)
    return DATASET_PATH


def load_dataframe(drop_id=False):
    """Read the saved dataset. Returns None if nothing has been uploaded yet."""
    if not os.path.exists(DATASET_PATH):
        return None
    # skipinitialspace handles headers like "SepalLengthCm, SepalWidthCm, ..."
    df = pd.read_csv(DATASET_PATH, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    if drop_id and df.shape[1] > 1:
        df = df.iloc[:, 1:]          # drop the first column (an id)
    return df


def feature_target_split(df):
    """Convention from the assignment: last column is the target, rest are features."""
    feature_names = list(df.columns[:-1])
    target_name = df.columns[-1]
    X = df[feature_names].apply(pd.to_numeric, errors="coerce")
    y = df[target_name]
    mask = X.notna().all(axis=1)     # drop rows with unparseable features
    return X[mask], y[mask], feature_names


def detect_problem_type(y):
    """Guess whether the target is a classification or a regression problem."""
    if y.dtype == object or str(y.dtype).startswith("category"):
        return "classification"
    values = pd.to_numeric(y, errors="coerce").dropna()
    looks_integer = np.allclose(values % 1, 0)
    if looks_integer and values.nunique() <= 20:
        return "classification"
    return "regression"


# ----------------------------------------------------------------------------
# Model registry: each model exposes ONE hyperparameter to tune
# ----------------------------------------------------------------------------
MODELS = {
    # --- classification ---
    "knn": {
        "label": "k-Nearest Neighbours",
        "problem": "classification",
        "estimator": KNeighborsClassifier,
        "param_name": "n_neighbors",
        "param_values": [1, 3, 5, 7, 9, 11, 15, 21],
        "scale": True,
    },
    "tree": {
        "label": "Decision Tree",
        "problem": "classification",
        "estimator": DecisionTreeClassifier,
        "param_name": "max_depth",
        "param_values": [1, 2, 3, 4, 5, 7, 10, 15],
        "scale": False,
    },
    "forest": {
        "label": "Random Forest",
        "problem": "classification",
        "estimator": RandomForestClassifier,
        "param_name": "n_estimators",
        "param_values": [10, 25, 50, 100, 200],
        "scale": False,
    },
    "logreg": {
        "label": "Logistic Regression",
        "problem": "classification",
        "estimator": LogisticRegression,
        "param_name": "C",
        "param_values": [0.01, 0.1, 1, 10, 100],
        "scale": True,
        "extra": {"max_iter": 1000},
    },
    # --- regression ---
    "knn_reg": {
        "label": "k-Nearest Neighbours (regression)",
        "problem": "regression",
        "estimator": KNeighborsRegressor,
        "param_name": "n_neighbors",
        "param_values": [1, 3, 5, 7, 9, 11, 15, 21],
        "scale": True,
    },
    "tree_reg": {
        "label": "Decision Tree (regression)",
        "problem": "regression",
        "estimator": DecisionTreeRegressor,
        "param_name": "max_depth",
        "param_values": [1, 2, 3, 4, 5, 7, 10, 15],
        "scale": False,
    },
    "forest_reg": {
        "label": "Random Forest (regression)",
        "problem": "regression",
        "estimator": RandomForestRegressor,
        "param_name": "n_estimators",
        "param_values": [10, 25, 50, 100, 200],
        "scale": False,
    },
    "ridge": {
        "label": "Ridge Regression",
        "problem": "regression",
        "estimator": Ridge,
        "param_name": "alpha",
        "param_values": [0.01, 0.1, 1, 10, 100],
        "scale": True,
    },
}


def grouped_models():
    """Models split by problem type, for the <optgroup> selector in the template."""
    groups = {"classification": [], "regression": []}
    for key, spec in MODELS.items():
        groups[spec["problem"]].append({"key": key, "label": spec["label"]})
    return groups


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def make_scatter(df, x_col, y_col, problem_type):
    """Scatter plot of two features, coloured by the target. Saved to media/."""
    _ensure_dir()
    target_name = df.columns[-1]
    y = df[target_name]

    plt.figure(figsize=(7, 5))
    if problem_type == "classification":
        for cls in pd.unique(y):
            mask = (y == cls)
            plt.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                        label=str(cls), alpha=0.7, edgecolor="k", linewidth=0.3)
        plt.legend(title=target_name)
    else:
        sc = plt.scatter(df[x_col], df[y_col], c=pd.to_numeric(y, errors="coerce"),
                         cmap="viridis", alpha=0.8, edgecolor="k", linewidth=0.3)
        plt.colorbar(sc, label=target_name)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "scatter.png"))
    plt.close()
    return settings.MEDIA_URL + "project1/scatter.png"


def _plot_scores(spec, train_scores, test_scores, metric):
    """Plot train/test score against each hyperparameter value."""
    _ensure_dir()
    values = spec["param_values"]
    positions = range(len(values))

    plt.figure(figsize=(7, 5))
    plt.plot(positions, train_scores, "o-", label="Train")
    plt.plot(positions, test_scores, "s-", label="Test")
    plt.xticks(list(positions), [str(v) for v in values])
    plt.xlabel(spec["param_name"])
    plt.ylabel(metric)
    plt.title(f"{spec['label']}: {metric} vs {spec['param_name']}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "training.png"))
    plt.close()
    return settings.MEDIA_URL + "project1/training.png"


# ----------------------------------------------------------------------------
# The training pipeline (Task 4)
# ----------------------------------------------------------------------------
def run_training(df, model_key, test_size):
    """
    Train `model_key` for every value of its hyperparameter:
      1. split into train / test
      2. (optionally) scale the features
      3. fit + score for each hyperparameter value
      4. plot the score curve and report the best value
    """
    spec = MODELS[model_key]
    is_clf = spec["problem"] == "classification"

    X, y, _ = feature_target_split(df)
    X, y = X.values, y.values

    # 1. train / test split (stratified for classification when possible)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if is_clf else None,
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42)

    # 2. scaling (only for distance / regularisation based models)
    if spec.get("scale"):
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    metric = "Accuracy" if is_clf else "R\u00b2 score"
    score = accuracy_score if is_clf else r2_score

    rows, train_scores, test_scores, best = [], [], [], None

    # 3. train one model per hyperparameter value
    for value in spec["param_values"]:
        kwargs = {spec["param_name"]: value}
        kwargs.update(spec.get("extra", {}))
        model = spec["estimator"](**kwargs)
        model.fit(X_tr, y_tr)

        tr = round(float(score(y_tr, model.predict(X_tr))), 4)
        te = round(float(score(y_te, model.predict(X_te))), 4)

        rows.append({"value": value, "train": tr, "test": te})
        train_scores.append(tr)
        test_scores.append(te)
        if best is None or te > best["test"]:
            best = {"value": value, "train": tr, "test": te}

    # 4. score curve
    plot_url = _plot_scores(spec, train_scores, test_scores, metric)

    return {
        "model_label": spec["label"],
        "param_name": spec["param_name"],
        "metric": metric,
        "rows": rows,
        "best": best,
        "plot_url": plot_url,
        "n_train": len(y_tr),
        "n_test": len(y_te),
    }
