#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function Time Predictor (stylish version)
----------------------------------------
- Reads data/data/functions_time.csv
- Builds a text -> TF-IDF (char 3–5) -> SGDRegressor pipeline
- 5-fold CV (R^2), compares to Dummy baseline
- Trains final model and predicts for two example functions
- Outputs compact, readable panels
"""
import os
import sys
import textwrap
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor

# -----------------------------
# Config
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "functions_time.csv")
RANDOM_STATE = 42

SAMPLES_TO_PREDICT = [
    ("optimize_cart_checkout", "Improves checkout speed by reducing API calls and optimizing queries"),
    ("login", "login user with email and password, returns auth token"),
]

# -----------------------------
# Pretty printing helpers
# -----------------------------
def panel(title: str, body_lines, width: int = 80):
    line = "─" * (width - 2)
    print(f"┌{line}┐")
    title_text = f"  {title.strip()}"
    print(f"│{title_text:<{width-2}}│")
    print(f"├{line}┤")
    if isinstance(body_lines, str):
        body_lines = body_lines.splitlines()
    for bl in body_lines:
        for line_part in textwrap.wrap(bl, width=width-4) or [""]:
            print(f"│  {line_part:<{width-4}}│")
    print(f"└{line}┘")

def fmt_scores(scores):
    return ", ".join(f"{s:.3f}" for s in scores)

# -----------------------------
# Load data
# -----------------------------
if not os.path.exists(DATA_PATH):
    panel("Error", [
        "CSV not found.",
        f"Expected at: {DATA_PATH}",
        "Make sure the dataset exists with columns: function_name, description, time_spent_hours"
    ])
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
if not {"function_name", "description", "time_spent_hours"}.issubset(df.columns):
    panel("Error", [
        "CSV missing required columns.",
        "Required: function_name, description, time_spent_hours"
    ])
    sys.exit(1)

df["combined_text"] = df["function_name"].astype(str) + " " + df["description"].astype(str)
X_text = df["combined_text"]
y = df["time_spent_hours"].astype(float)

# -----------------------------
# Build models
# -----------------------------
# Baseline (predict-mean) inside a pipeline so cross_val_score can run
baseline = make_pipeline(
    TfidfVectorizer(analyzer="word", ngram_range=(1,1)),
    DummyRegressor(strategy="mean")
)

# Improved model: character n-grams help with code-like tokens & typos
model = make_pipeline(
    TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2),
    SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-4,
                 random_state=RANDOM_STATE, max_iter=2000)
)

# -----------------------------
# Evaluate
# -----------------------------
base_scores = cross_val_score(baseline, X_text, y, cv=5, scoring="r2")
model_scores = cross_val_score(model, X_text, y, cv=5, scoring="r2")

panel("Dataset", [
    f"Rows: {len(df)}",
    "Text field: function_name + description",
])

panel("Cross-Validation (5-fold, R²)", [
    f"Baseline (Dummy mean): [{fmt_scores(base_scores)}] | avg: {base_scores.mean():.3f}",
    f"Model (char 3–5 + SGD): [{fmt_scores(model_scores)}] | avg: {model_scores.mean():.3f}",
])

# -----------------------------
# Train final & predict
# -----------------------------
model.fit(X_text, y)

pred_lines = []
for name, desc in SAMPLES_TO_PREDICT:
    text = name + " " + desc
    pred = float(model.predict([text])[0])
    pred_lines.append(f"{name:>24}: {pred:5.2f} hours")

panel("Predictions", pred_lines)

# -----------------------------
# Tips
# -----------------------------
panel("Notes", [
    "• R² near 0 means weak predictive signal. Add more rows & richer descriptions.",
    "• Character n-grams are often effective on short, code-like tokens.",
    "• Keep comparing to the Dummy baseline to ensure real gains.",
])
