"""
ml_basics/decision_tree_titanic.py

Train a Decision Tree classifier on the Titanic dataset, compare to Logistic Regression,
visualize the tree, run a tiny hyperparameter experiment, and optionally explain a single
prediction in "simple" or "engineer" mode (rule-based).

Usage:
    python ml_basics/decision_tree_titanic.py --csv titanic.csv
    # optional flags:
    --reports_dir reports
    --explain_row 5 --mode simple

Assumptions:
- A file named "titanic.csv" exists at the given path (default tries: provided --csv,
  then ./titanic.csv, then ../titanic.csv).
- Columns include at least: ["Survived","Pclass","Sex","Age","Fare","Embarked"].
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


RANDOM_STATE = 0

FEATURES = ["Pclass","Sex","Age","Fare","Embarked"]
TARGET = "Survived"

NUMERIC = ["Pclass","Age","Fare"]
CATEGORICAL = ["Sex","Embarked"]


def find_csv_path(cli_path: Optional[str]) -> Path:
    """Resolve a CSV path sensibly."""
    if cli_path and Path(cli_path).exists():
        return Path(cli_path)

    candidates = [Path("titanic.csv"), Path("../titanic.csv"), Path("data/titanic.csv"), Path("../data/titanic.csv")]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find titanic.csv. Pass --csv /path/to/titanic.csv")


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Keep the feature set we care about
    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    df = df[FEATURES + [TARGET]].copy()

    # Minimal cleanup: fill simple missing values
    if df["Age"].isna().any():
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if df["Fare"].isna().any():
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if df["Embarked"].isna().any():
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    return df


def make_preprocess() -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough"
    )


def train_tree_pipeline(X_tr, y_tr) -> Pipeline:
    preprocess = make_preprocess()
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    clf = Pipeline([("prep", preprocess), ("model", model)])
    clf.fit(X_tr, y_tr)
    return clf


def train_logreg_pipeline(X_tr, y_tr) -> Pipeline:
    preprocess = make_preprocess()
    # liblinear works well for small datasets / binary classification
    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    clf = Pipeline([("prep", preprocess), ("model", model)])
    clf.fit(X_tr, y_tr)
    return clf


def get_feature_names_from_preprocess(prep: ColumnTransformer) -> List[str]:
    # works for sklearn >= 1.0
    try:
        return list(prep.get_feature_names_out())
    except Exception:
        # Fallback best-effort
        cat_names = []
        for name, trans, cols in prep.transformers_:
            if name == "cat":
                # OneHotEncoder
                try:
                    cat_names = list(trans.get_feature_names_out(cols))
                except Exception:
                    cat_names = [f"{name}__{c}" for c in cols]
        return cat_names + NUMERIC


def save_tree_plots_and_text(clf: Pipeline, reports_dir: Path) -> Tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    model: DecisionTreeClassifier = clf.named_steps["model"]
    prep: ColumnTransformer = clf.named_steps["prep"]
    feat_names = get_feature_names_from_preprocess(prep)

    # Top of the tree figure (depth=3 for readability)
    fig = plt.figure(figsize=(16, 10))
    plot_tree(
        model,
        feature_names=feat_names,
        class_names=["died","survived"],
        filled=True,
        max_depth=3,
        impurity=True
    )
    top_png = reports_dir / "tree_top_depth3.png"
    fig.tight_layout()
    fig.savefig(top_png, dpi=150)
    plt.close(fig)

    # Text export of the top of the tree
    from sklearn.tree import export_text
    txt = export_text(model, feature_names=feat_names, max_depth=3, decimals=2)
    top_txt = reports_dir / "tree_top_depth3.txt"
    with open(top_txt, "w", encoding="utf-8") as f:
        f.write(txt)

    return top_png, top_txt


def mini_experiment_grid(X_tr, y_tr, X_val, y_val, reports_dir: Path) -> pd.DataFrame:
    combos = []
    for max_depth in [2, 4, None]:
        for min_samples_split in [2, 20]:
            preprocess = make_preprocess()
            model = DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                max_depth=max_depth,
                min_samples_split=min_samples_split
            )
            clf = Pipeline([("prep", preprocess), ("model", model)])
            clf.fit(X_tr, y_tr)
            # scores
            tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
            val_acc = accuracy_score(y_val, clf.predict(X_val))
            combos.append({
                "max_depth": max_depth if max_depth is not None else "None",
                "min_samples_split": min_samples_split,
                "train_acc": round(tr_acc, 4),
                "val_acc": round(val_acc, 4),
            })

    df = pd.DataFrame(combos).sort_values("val_acc", ascending=False).reset_index(drop=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "decision_tree_grid.csv"
    df.to_csv(out_csv, index=False)
    return df


def explain_instance(
    clf: Pipeline,
    x_row: pd.DataFrame,
    mode: str = "simple"
) -> Tuple[str, List[str]]:
    """
    Return (prediction_str, rules) for a single-row DataFrame x_row.
    mode: "simple" or "engineer"
    """
    assert len(x_row) == 1, "x_row must contain exactly one row"

    prep: ColumnTransformer = clf.named_steps["prep"]
    model: DecisionTreeClassifier = clf.named_steps["model"]

    # Transform row to model's feature space
    Xt = prep.transform(x_row)  # shape (1, n_features)
    feat_names = get_feature_names_from_preprocess(prep)

    tree = model.tree_
    node = 0
    rules = []
    from sklearn.tree import _tree

    # Helper to prettify feature names
    def pretty_name(name: str) -> str:
        # Usually looks like 'cat__Sex_female' or 'cat__Embarked_S'
        if name.startswith("cat__"):
            rest = name[5:]
            if rest.startswith("Sex_"):
                v = rest.split("_", 1)[1]
                return f"Sex = {v}"
            if rest.startswith("Embarked_"):
                v = rest.split("_", 1)[1]
                return f"Embarked = {v}"
        # Otherwise it might be the numeric ones
        return name.replace("remainder__", "")

    while tree.feature[node] != _tree.TREE_UNDEFINED:
        feat_idx = tree.feature[node]
        thresh = tree.threshold[node]
        fname = feat_names[feat_idx]
        xval = Xt[0, feat_idx]

        go_left = xval <= thresh
        # Build rule text
        if mode == "simple":
            # Special-case binary one-hot splits (~0.5 threshold) to ~= equality
            if fname.startswith("cat__") and 0.49 <= thresh <= 0.51:
                # if xval <= 0.5 means xval is 0 -> NOT that category
                positive = pretty_name(fname)
                if go_left:
                    rule = f"NOT ({positive})"
                else:
                    rule = f"{positive}"
            else:
                op = "<=" if go_left else ">"
                rule = f"{pretty_name(fname)} {op} {thresh:.2f}"
        else:  # engineer
            op = "<=" if go_left else ">"
            rule = f"{fname} (value {xval:.3f}) {op} threshold {thresh:.3f}"

        rules.append(rule)
        node = tree.children_left[node] if go_left else tree.children_right[node]

    # At leaf: compute predicted class + prob
    proba = model.predict_proba(prep.transform(x_row))[0]
    pred_class = int(proba[1] >= 0.5)
    confidence = proba[pred_class]
    pred_label = "Survived" if pred_class == 1 else "Died"
    pred_str = f"Prediction: {pred_label} ({confidence*100:.1f}% chance)"

    return pred_str, rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to titanic.csv")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Directory to save plots/text/csv")
    parser.add_argument("--explain_row", type=int, default=None, help="Row index (from validation set) to explain")
    parser.add_argument("--mode", type=str, default="simple", choices=["simple","engineer"], help="Explanation mode")
    args = parser.parse_args()

    csv_path = find_csv_path(args.csv)
    reports_dir = Path(args.reports_dir)

    print(f"Using CSV at: {csv_path.resolve()}")

    # Load + clean
    df = load_and_clean(csv_path)

    X = df[FEATURES]
    y = df[TARGET]

    # Train/validation split (stratified)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Train Decision Tree
    tree_clf = train_tree_pipeline(X_tr, y_tr)
    val_pred_tree = tree_clf.predict(X_val)
    val_acc_tree = accuracy_score(y_val, val_pred_tree)
    print(f"[DecisionTree] Validation accuracy: {val_acc_tree:.4f}")

    # Compare to Logistic Regression (baseline from yesterday)
    logreg_clf = train_logreg_pipeline(X_tr, y_tr)
    val_pred_lr = logreg_clf.predict(X_val)
    val_acc_lr = accuracy_score(y_val, val_pred_lr)
    print(f"[LogisticRegression] Validation accuracy: {val_acc_lr:.4f}")

    delta = val_acc_tree - val_acc_lr
    better = "Decision Tree" if delta > 0 else ("Logistic Regression" if delta < 0 else "Tie")
    print(f"[Compare] {better} by {abs(delta):.4f} accuracy")

    # Visualize + save the top of the tree and a text version
    top_png, top_txt = save_tree_plots_and_text(tree_clf, reports_dir)
    print(f"Saved: {top_png}")
    print(f"Saved: {top_txt}")

    # Mini-experiment (depth & min_samples_split)
    grid_df = mini_experiment_grid(X_tr, y_tr, X_val, y_val, reports_dir)
    print("\nMini-experiment (sorted by validation accuracy):")
    print(grid_df.to_string(index=False))
    print(f"Saved grid to: {Path(args.reports_dir) / 'decision_tree_grid.csv'}")

    # Optional: explain a single validation row
    if args.explain_row is not None:
        if args.explain_row < 0 or args.explain_row >= len(X_val):
            print(f"[Explain] Row index out of range. Valid 0..{len(X_val)-1}")
        else:
            x_row = X_val.iloc[[args.explain_row]]
            pred_str, rules = explain_instance(tree_clf, x_row, mode=args.mode)
            print("\n=== Local Explanation ===")
            print(pred_str)
            print("Rules followed:")
            for r in rules:
                print(f"- {r}")

    print("\nDone. Open the image/text in the reports/ folder to read the tree rules.")
    print("Tip: try --explain_row 0 --mode simple  (or --mode engineer) to see per-passenger rules.")
    

if __name__ == "__main__":
    main()
