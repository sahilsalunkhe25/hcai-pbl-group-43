"""
Global model-agnostic feature effects (Task 5): PDP and ALE, computed by hand.

Both describe how a single numerical feature affects the predicted probability
of each species (three curves per plot).

* PDP  (Partial Dependence): average the model's predicted probability over the
  whole dataset while forcing the chosen feature to each grid value.

* ALE  (Accumulated Local Effects): accumulate *local* changes in prediction,
  which removes the bias PDP suffers from when features are correlated.

  - For the **decision tree** the prediction is piecewise-constant, so its
    derivative is (almost everywhere) zero; we must use a *discretisation*:
    finite differences across quantile bin edges.
  - For **logistic regression** the softmax probability is differentiable, so we
    use the *exact* analytic partial derivative and integrate it.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from django.conf import settings

from . import ml


def _classes(pipe):
    return list(pipe.named_steps["clf"].classes_)


# ----------------------------------------------------------------------------
# PDP  (model-agnostic)
# ----------------------------------------------------------------------------
def pdp(pipe, feature, grid_size=40):
    """Partial dependence of each class probability on ``feature``."""
    df = ml.get_data()["df"]
    X = df[ml.FEATURES].copy()
    lo, hi = np.percentile(X[feature].astype(float), [2, 98])
    grid = np.linspace(lo, hi, grid_size)

    classes = _classes(pipe)
    curves = {c: np.zeros(grid_size) for c in classes}

    for i, v in enumerate(grid):
        Xv = X.copy()
        Xv[feature] = v
        proba = pipe.predict_proba(Xv)          # (n_samples, n_classes)
        mean = proba.mean(axis=0)
        for j, c in enumerate(classes):
            curves[c][i] = mean[j]
    return grid, curves


# ----------------------------------------------------------------------------
# ALE via discretisation  (used for the tree; works for any model)
# ----------------------------------------------------------------------------
def ale_discretized(pipe, feature, n_bins=10):
    """Finite-difference ALE: local effects across quantile bins, accumulated."""
    df = ml.get_data()["df"]
    X = df[ml.FEATURES].copy()
    x = X[feature].astype(float).to_numpy()

    # quantile bin edges (drop duplicates for safety)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    n_bins = len(edges) - 1
    classes = _classes(pipe)

    # local effect per bin = mean over points in bin of [f(upper) - f(lower)]
    local = {c: np.zeros(n_bins) for c in classes}
    counts = np.zeros(n_bins)

    # assign each point to a bin
    idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, n_bins - 1)

    for b in range(n_bins):
        in_bin = np.where(idx == b)[0]
        counts[b] = len(in_bin)
        if len(in_bin) == 0:
            continue
        Xb = X.iloc[in_bin].copy()
        lower, upper = Xb.copy(), Xb.copy()
        lower[feature] = edges[b]
        upper[feature] = edges[b + 1]
        diff = pipe.predict_proba(upper) - pipe.predict_proba(lower)
        mean_diff = diff.mean(axis=0)
        for j, c in enumerate(classes):
            local[c][b] = mean_diff[j]

    # accumulate, then centre (mean effect over the data = 0)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    curves = {}
    total = counts.sum()
    for c in classes:
        acc = np.concatenate([[0.0], np.cumsum(local[c])])    # at bin edges
        acc_centers = (acc[:-1] + acc[1:]) / 2.0
        weighted_mean = np.sum(counts * acc_centers) / total if total else 0.0
        curves[c] = acc_centers - weighted_mean
    return bin_centers, curves


# ----------------------------------------------------------------------------
# ALE via exact derivative  (logistic regression only)
# ----------------------------------------------------------------------------
def ale_exact_logreg(pipe, feature, grid_size=40):
    """Exact ALE for the softmax model using the analytic probability gradient.

    For standardised input z = (x - mu)/sigma and logits f_c = w_c . z + b_c,
        dp_c/dx_f = p_c * (w_{c,f} - sum_k p_k w_{k,f}) / sigma_f .
    We average this expected derivative over the data at each grid value and
    integrate (trapezoid) along the feature axis, then centre.
    """
    df = ml.get_data()["df"]
    X = df[ml.FEATURES].copy()

    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["pre"].named_transformers_["num"]
    f_idx = ml.NUMERIC_FEATURES.index(feature)        # column in the num block
    sigma_f = scaler.scale_[f_idx]
    W = clf.coef_                                       # (n_classes, n_features)
    w_f = W[:, f_idx]                                   # weight on this feature
    classes = _classes(pipe)

    lo, hi = np.percentile(X[feature].astype(float), [2, 98])
    grid = np.linspace(lo, hi, grid_size)

    # expected derivative at each grid value
    deriv = {c: np.zeros(grid_size) for c in classes}
    for i, v in enumerate(grid):
        Xv = X.copy()
        Xv[feature] = v
        P = pipe.predict_proba(Xv)                      # (n, n_classes)
        wbar = P @ w_f                                  # sum_k p_k w_{k,f}, per row
        for j, c in enumerate(classes):
            dpc = P[:, j] * (w_f[j] - wbar) / sigma_f
            deriv[c][i] = dpc.mean()

    # integrate then centre
    curves = {}
    x = X[feature].astype(float).to_numpy()
    for j, c in enumerate(classes):
        acc = np.concatenate([[0.0], np.cumsum(
            (deriv[c][1:] + deriv[c][:-1]) / 2.0 * np.diff(grid))])
        # centre over the data distribution of the feature
        interp = np.interp(x, grid, acc)
        curves[c] = acc - interp.mean()
    return grid, curves


# ----------------------------------------------------------------------------
# Dispatch + plotting
# ----------------------------------------------------------------------------
def feature_effects(pipe, model_class, feature):
    """Compute PDP and ALE; ALE method depends on the model class."""
    pdp_grid, pdp_curves = pdp(pipe, feature)
    if model_class == "logreg":
        ale_grid, ale_curves = ale_exact_logreg(pipe, feature)
        ale_method = "exact analytic derivative"
    else:
        ale_grid, ale_curves = ale_discretized(pipe, feature)
        ale_method = "discretisation (finite differences)"
    return {
        "pdp": (pdp_grid, pdp_curves),
        "ale": (ale_grid, ale_curves),
        "ale_method": ale_method,
    }


def render_effects(pipe, model_class, feature):
    """Draw PDP (left) and ALE (right) and return the image URL + ALE method."""
    os.makedirs(ml.DATA_DIR, exist_ok=True)
    eff = feature_effects(pipe, model_class, feature)
    pdp_grid, pdp_curves = eff["pdp"]
    ale_grid, ale_curves = eff["ale"]

    fig, (axp, axa) = plt.subplots(1, 2, figsize=(12, 4.6))
    for c in pdp_curves:
        axp.plot(pdp_grid, pdp_curves[c], "-", label=str(c))
    axp.set_title(f"PDP — {feature}")
    axp.set_xlabel(feature)
    axp.set_ylabel("avg. P(species)")
    axp.legend(title="species")
    axp.grid(alpha=0.3)

    for c in ale_curves:
        axa.plot(ale_grid, ale_curves[c], "-", label=str(c))
    axa.axhline(0, color="#999", lw=0.8)
    axa.set_title(f"ALE — {feature}  ({eff['ale_method']})")
    axa.set_xlabel(feature)
    axa.set_ylabel("centred effect on P(species)")
    axa.legend(title="species")
    axa.grid(alpha=0.3)

    fig.tight_layout()
    fname = "effects.png"
    fig.savefig(os.path.join(ml.DATA_DIR, fname), dpi=110)
    plt.close(fig)
    return settings.MEDIA_URL + "project2/" + fname, eff["ale_method"]
