"""
Counterfactual explanations (Task 4).

Given a query point ``x`` and a desired target class, we:

1. randomly sample ``N`` points locally around ``x``;
2. keep those the (selected) model predicts as the target class;
3. rank them by their MAD-weighted L1 distance to ``x`` (lecture 5);
4. return the best ``k``.

Feature types are handled differently when noising:

* continuous features -> additive Gaussian noise scaled by the feature's MAD;
* categorical / binary  -> randomly switched to another category with some prob.

If nothing is found, we iteratively grow ``N`` and the sampling spread.
"""

import numpy as np
import pandas as pd

from . import ml


def _mad(series):
    """Median absolute deviation, with a safe fallback for (near-)constant cols."""
    values = series.to_numpy(dtype=float)
    mad = np.median(np.abs(values - np.median(values)))
    if mad <= 1e-9:                      # avoid divide-by-zero
        std = values.std()
        return std if std > 1e-9 else 1.0
    return float(mad)


def _feature_meta():
    """Per-feature scale (MAD) for numerics and the value domain for categoricals."""
    df = ml.get_data()["df"]
    mads = {f: _mad(df[f]) for f in ml.NUMERIC_FEATURES}
    domains = {f: sorted(df[f].astype(str).unique().tolist())
               for f in ml.CATEGORICAL_FEATURES}
    return mads, domains


def mad_l1_distance(x, candidates, mads):
    """MAD-weighted L1 distance from row ``x`` to each row in ``candidates``.

    Continuous: |x_j - x'_j| / MAD_j.  Categorical: 1 if different, else 0.
    """
    dist = np.zeros(len(candidates), dtype=float)
    for f in ml.NUMERIC_FEATURES:
        dist += np.abs(candidates[f].to_numpy(float) - float(x[f])) / mads[f]
    for f in ml.CATEGORICAL_FEATURES:
        dist += (candidates[f].astype(str).to_numpy() != str(x[f])).astype(float)
    return dist


def _sample_around(x, n, mads, domains, spread, flip_prob, rng):
    """Draw ``n`` perturbations of ``x`` (returns a DataFrame in feature order)."""
    data = {}
    # continuous: Gaussian noise scaled by MAD * spread
    for f in ml.NUMERIC_FEATURES:
        data[f] = float(x[f]) + rng.normal(0.0, spread * mads[f], size=n)
    # categorical: with prob flip_prob, switch to a random *other* category
    for f in ml.CATEGORICAL_FEATURES:
        col = np.array([str(x[f])] * n, dtype=object)
        flip = rng.random(n) < flip_prob
        choices = domains[f]
        for i in np.where(flip)[0]:
            alts = [c for c in choices if c != str(x[f])] or choices
            col[i] = rng.choice(alts)
        data[f] = col
    return pd.DataFrame(data, columns=ml.FEATURES)


def generate(pipe, x, target_label, k=5, seed=0):
    """Return up to ``k`` counterfactuals for ``x`` predicted as ``target_label``.

    Iteratively grows ``N`` and the sampling spread until at least ``k`` valid
    counterfactuals are found (or we give up after a few rounds).
    """
    rng = np.random.default_rng(seed)
    mads, domains = _feature_meta()

    n = 1000
    spread = 0.5
    flip_prob = 0.3
    found = pd.DataFrame()

    rounds = 0
    while rounds < 6:
        rounds += 1
        cand = _sample_around(x, n, mads, domains, spread, flip_prob, rng)
        preds = pipe.predict(cand)
        hits = cand[preds == target_label]
        if len(hits) > 0:
            found = pd.concat([found, hits], ignore_index=True)
        if len(found) >= max(k, 20):
            break
        # nothing (or too little) yet -> cast a wider net
        n *= 2
        spread *= 1.5
        flip_prob = min(0.6, flip_prob + 0.1)

    if len(found) == 0:
        return [], rounds

    found = found.drop_duplicates().reset_index(drop=True)
    found["__dist"] = mad_l1_distance(x, found, mads)
    found = found.sort_values("__dist").head(k).reset_index(drop=True)

    # Build display rows: value per feature + which features changed vs x.
    results = []
    for _, row in found.iterrows():
        changes = {}
        for f in ml.FEATURES:
            if f in ml.NUMERIC_FEATURES:
                changed = abs(float(row[f]) - float(x[f])) > 1e-6
                shown = round(float(row[f]), 2)
            else:
                changed = str(row[f]) != str(x[f])
                shown = str(row[f])
            changes[f] = {"value": shown, "changed": changed}
        results.append({"distance": round(float(row["__dist"]), 3), "features": changes})
    return results, rounds
