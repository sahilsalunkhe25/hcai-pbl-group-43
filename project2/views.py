from django.shortcuts import render

from . import explain


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def index(request):
    """
    Single-page dashboard: model class + regularization (Tasks 1-3),
    counterfactual explanations (Task 4), feature effect plots (Task 5).
    Every control is a GET param so the page stays on one bookmarkable URL.
    """
    model_class = request.GET.get("model", "tree")
    if model_class not in explain.MODEL_LABELS:
        model_class = "tree"

    try:
        lam = _clamp(float(request.GET.get("lam", 0.0)), 0.0, explain.LAMBDA_MAX)
    except ValueError:
        lam = 0.0

    feature = request.GET.get("feature", explain.NUMERIC[0])
    if feature not in explain.NUMERIC:
        feature = explain.NUMERIC[0]

    grid = explain.get_grid(model_class)
    best = explain.select_best(grid, lam)
    model_plot_url = explain.plot_model(model_class, best)

    species_list = explain.get_species_list()
    example_choices = explain.get_example_choices()

    example_idx = None
    raw_example = request.GET.get("example")
    if raw_example:
        try:
            example_idx = int(raw_example)
        except ValueError:
            example_idx = None

    target_class = request.GET.get("target") or None
    if target_class not in species_list:
        target_class = None

    cf_result = None
    if example_idx is not None and target_class is not None:
        example_row = explain.get_example_row(example_idx)
        if example_row is not None:
            cf_result = explain.generate_counterfactuals(example_row, target_class, best["pipeline"])

    pdp_grid, pdp_values, classes = explain.compute_pdp(best["pipeline"], feature)
    ale_edges, ale_values, _ = explain.compute_ale(best["pipeline"], feature, model_class)
    effects_plot_url = explain.plot_pdp_ale(feature, pdp_grid, pdp_values, ale_edges, ale_values, classes)

    context = {
        "model_class": model_class,
        "model_label": explain.MODEL_LABELS[model_class],
        "complexity_label": explain.COMPLEXITY_LABELS[model_class],
        "lam": lam,
        "lam_max": explain.LAMBDA_MAX,
        "lam_step": explain.LAMBDA_STEP,
        "best": best,
        "model_plot_url": model_plot_url,
        "species_list": species_list,
        "example_choices": example_choices,
        "selected_example": example_idx,
        "selected_example_value": "" if example_idx is None else str(example_idx),
        "selected_target": target_class,
        "selected_target_value": target_class or "",
        "cf_result": cf_result,
        "numeric_features": explain.NUMERIC,
        "selected_feature": feature,
        "effects_plot_url": effects_plot_url,
    }
    return render(request, "project2/index.html", context)
