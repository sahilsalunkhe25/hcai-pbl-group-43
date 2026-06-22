"""
Project 2 (Explainability) — a single linked interface.

All sections (model display, counterfactuals, feature effects) are driven by the
same two controls read from the query string:

* ``model_class`` : "tree" or "logreg"
* ``lam``         : the regularisation slider lambda

so the user freely picks the model family and sparsity, and every region reacts.
"""

from django.shortcuts import render

from . import ml
from . import counterfactuals as cf
from . import explain


def _read_controls(request):
    """Parse the shared model-selection controls from the query string."""
    model_class = request.GET.get("model_class", "tree")
    if model_class not in ("tree", "logreg"):
        model_class = "tree"
    try:
        lam = float(request.GET.get("lam", 0.0))
    except (TypeError, ValueError):
        lam = 0.0
    return model_class, lam


def index(request):
    model_class, lam = _read_controls(request)
    best, family = ml.select_model(model_class, lam)
    model_img = ml.render_model(best["pipe"], model_class)

    data = ml.get_data()
    classes = data["classes"]
    df = data["df"]

    # ---- Counterfactual region (Task 4) ----
    n_rows = len(df)
    try:
        cf_index = int(request.GET.get("cf_index", 0))
    except (TypeError, ValueError):
        cf_index = 0
    cf_index = max(0, min(cf_index, n_rows - 1))

    x_row = df.iloc[cf_index]
    x_features = x_row[ml.FEATURES]
    x_actual = x_row[ml.TARGET]
    x_pred = best["pipe"].predict(x_features.to_frame().T)[0]

    cf_target = request.GET.get("cf_target") or next(
        (c for c in classes if c != x_pred), classes[0])

    cf_results, cf_rounds = [], 0
    if request.GET.get("cf_run"):
        cf_results, cf_rounds = cf.generate(
            best["pipe"], x_features, cf_target, k=5)

    # ---- Feature effect plots (Task 5) ----
    fe_feature = request.GET.get("fe_feature") or ml.NUMERIC_FEATURES[0]
    if fe_feature not in ml.NUMERIC_FEATURES:
        fe_feature = ml.NUMERIC_FEATURES[0]
    fe_img, fe_method = explain.render_effects(best["pipe"], model_class, fe_feature)

    context = {
        "model_class": model_class,
        "lam": lam,
        "best": best,
        "family": family,
        "model_img": model_img,
        "classes": classes,
        "features": ml.FEATURES,
        # counterfactual context
        "n_rows": n_rows,
        "cf_index": cf_index,
        "cf_x": {f: (round(float(x_features[f]), 2) if f in ml.NUMERIC_FEATURES
                     else str(x_features[f])) for f in ml.FEATURES},
        "cf_actual": x_actual,
        "cf_pred": x_pred,
        "cf_target": cf_target,
        "cf_results": cf_results,
        "cf_rounds": cf_rounds,
        "cf_ran": bool(request.GET.get("cf_run")),
        # feature effects
        "numeric_features": ml.NUMERIC_FEATURES,
        "fe_feature": fe_feature,
        "fe_img": fe_img,
        "fe_method": fe_method,
    }
    return render(request, "project2/index.html", context)
