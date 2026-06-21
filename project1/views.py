from django.shortcuts import render, redirect

from .forms import CSVUploadForm
from . import ml

# Test-set fractions offered to the user (percent).
TEST_SIZES = [10, 20, 30, 40]


def index(request):
    """Entry point: upload a CSV, then go to the explore page."""
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            ml.save_uploaded_dataset(request.FILES["file"])
            request.session["drop_id"] = form.cleaned_data["drop_first_column"]
            return redirect("project1:explore")
    else:
        form = CSVUploadForm()
    return render(request, "project1/index.html", {"form": form})


def explore(request):
    """Show a preview of the data and a scatter plot of two chosen features."""
    df = ml.load_dataframe(drop_id=request.session.get("drop_id", False))
    if df is None:
        return redirect("project1:index")

    feature_names = list(df.columns[:-1])
    target_name = df.columns[-1]
    problem_type = ml.detect_problem_type(df[target_name])

    # default to the first two features; let the user override via the form
    x_col = request.GET.get("x") or (feature_names[0] if feature_names else None)
    if len(feature_names) > 1:
        y_col = request.GET.get("y") or feature_names[1]
    else:
        y_col = request.GET.get("y") or x_col

    plot_url = ml.make_scatter(df, x_col, y_col, problem_type) if x_col and y_col else None

    context = {
        "n_rows": df.shape[0],
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "target_name": target_name,
        "problem_type": problem_type,
        "preview": df.head(10).to_html(classes="data-table", index=False),
        "x_col": x_col,
        "y_col": y_col,
        "plot_url": plot_url,
    }
    return render(request, "project1/explore.html", context)


def train(request):
    """Train the chosen model over its hyperparameter range and show the results."""
    df = ml.load_dataframe(drop_id=request.session.get("drop_id", False))
    if df is None:
        return redirect("project1:index")

    detected = ml.detect_problem_type(df[df.columns[-1]])
    selected = {"model": None, "test_size": 30}
    results = None

    if request.method == "POST":
        model_key = request.POST.get("model")
        test_size = int(request.POST.get("test_size", 30))
        selected = {"model": model_key, "test_size": test_size}
        if model_key in ml.MODELS:
            results = ml.run_training(df, model_key, test_size / 100.0)

    context = {
        "detected": detected,
        "models": ml.grouped_models(),
        "test_sizes": TEST_SIZES,
        "selected": selected,
        "results": results,
    }
    return render(request, "project1/train.html", context)
