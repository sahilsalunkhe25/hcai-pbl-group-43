# HCAI Project Based Learning — Group 43

Human-Centered AI (HCAI) project-based learning coursework, TUHH.

This repository will hold **all 5 projects** for the course, each as a Django app:

- **Project 1 — Supervised Learning:** upload a CSV, explore it visually, and train models across a range of hyperparameters.
- **Project 2 — Explainability:** interpretability, counterfactuals, and feature-effect plots on the Palmer Penguins dataset.

### Group 43

- Sahil Salunkhe
- Aditya Anil Gupta

---

## Project 1 — Interactive ML on CSV data

A Django app that lets you:

1. **Upload** a CSV dataset (optionally dropping the first column if it is an ID).
2. **Explore** the data — preview the first rows, see the number of rows/features, the detected problem type (classification vs. regression), and a scatter plot of any two features coloured by the target.
3. **Train** a model — pick a model, choose a train/test split, and the app fits it across a sweep of one hyperparameter, plots the train/test score curve, and reports the best value.

**Convention:** the **last column** of the CSV is treated as the target; all other columns are features.

### Available models

| Problem type   | Models                                                              | Tuned hyperparameter        |
| -------------- | ------------------------------------------------------------------ | --------------------------- |
| Classification | k-Nearest Neighbours, Decision Tree, Random Forest, Logistic Reg.  | `n_neighbors` / `max_depth` / `n_estimators` / `C` |
| Regression     | k-Nearest Neighbours, Decision Tree, Random Forest, Ridge          | `n_neighbors` / `max_depth` / `n_estimators` / `alpha` |

The problem type is detected automatically from the target column. Metrics: **accuracy** for classification, **R²** for regression.

---

---

## Project 2 — Explainability

A single **linked interface** on the Palmer Penguins dataset (predict `species`). Pick a
**model class** (decision tree / logistic regression) and a sparsity level **λ** via a
slider; every section reacts to that choice.

1. **Interpretability & complexity** (Tasks 1–3) — the selected model is shown (the tree
   itself, or the logistic-regression coefficient heatmap) with its **test accuracy** and
   complexity **Ω** (tree → number of leaves; logreg → number of non-zero L1 coefficients).
   The λ slider selects the family member maximising `acc_test − λ·Ω(f)`.
2. **Counterfactual explanations** (Task 4) — pick an example and a target species; the app
   samples locally around the point (Gaussian noise on numeric features, random category
   switches on categorical ones), keeps points the selected model predicts as the target,
   and ranks them by **MAD-weighted L1 distance**, widening the search iteratively if needed.
3. **Feature effect plots** (Task 5) — **PDP** and **ALE** (both implemented from scratch,
   no library) for each numerical feature, one curve per species. ALE uses the **exact
   analytic derivative** for logistic regression and **discretisation** (finite differences)
   for the decision tree.

Routes: `/project2/` (everything is driven by query-string controls on that one page).

---

## Tech stack

- **Python 3.10**, **Django 4.2**
- **pandas** / **numpy** for data handling
- **scikit-learn** for model training
- **matplotlib** (Agg backend) for server-side plot rendering
- **palmerpenguins** for the Project 2 dataset

---

## Setup & running

From the repository root:

```bash
# 1. Create and activate a virtual environment (one-time)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies (one-time)
pip install -r requirements.txt

# 3. Apply database migrations
python manage.py migrate

# 4. Start the development server
python manage.py runserver
```

Then open **http://127.0.0.1:8000/** in your browser.

After the first setup, you only need:

```bash
source .venv/bin/activate
python manage.py runserver
```

---

## Routes

| URL          | Description                                          |
| ------------ | --------------------------------------------------- |
| `/`          | Redirects to the home page                          |
| `/home/`     | Landing page                                        |
| `/project1/` | Project 1: CSV upload → explore → train workflow    |
| `/project2/` | Project 2: explainability interface (Palmer Penguins) |
| `/demos/`    | Demo views (CSV upload / plot generation)           |
| `/admin/`    | Django admin                                         |

---

## Project layout

```
pbl/         Django project settings & root URL config
home/         Landing page app
project1/     Project 1 app
  ├── views.py    Web layer (upload / explore / train views)
  ├── ml.py       Data handling, plotting, and the training pipeline
  └── forms.py    CSV upload form
demos/        Demo / scratch views
templates/    Shared templates
static/       Shared static assets
media/        Uploaded datasets and generated plots (runtime)
```

> **Note:** uploaded datasets and generated plots are written to `media/`. The local `.venv/` and `media/` artifacts should not be committed.
