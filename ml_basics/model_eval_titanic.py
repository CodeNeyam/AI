# ml_basics/model_eval_titanic.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

def load_data(csv_arg: str | None) -> pd.DataFrame:
    if csv_arg:
        path = Path(csv_arg)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        return pd.read_csv(path)

    # try common locations in your repo
    candidates = [
        Path("data_exploration/titanic.csv"),
        Path("ml_basics/titanic.csv"),
        Path("titanic.csv"),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("Could not find titanic.csv in expected locations. "
                            "Pass a path with --csv")

def build_preprocessor(df: pd.DataFrame):
    # Typical Titanic columns; adjust if yours differ
    num_features = ["Age", "SibSp", "Parch", "Fare"]
    cat_features = ["Pclass", "Sex", "Embarked"]

    # Keep only columns that actually exist in the CSV
    num_features = [c for c in num_features if c in df.columns]
    cat_features = [c for c in cat_features if c in df.columns]

    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features),
        ]
    )
    feature_list = num_features + cat_features
    return preprocessor, feature_list

def split_xy(df: pd.DataFrame):
    # Common target is Survived (0/1)
    if "Survived" not in df.columns:
        raise KeyError("Expected a 'Survived' column as target.")
    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])
    return X, y

def evaluate(model_name: str, y_true, y_pred, y_proba=None):
    print(f"\n=== {model_name} ===")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=3))
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    if y_proba is not None:
        try:
            print(f"ROC AUC  : {roc_auc_score(y_true, y_proba):.3f}")
        except ValueError:
            pass

def main():
    parser = argparse.ArgumentParser(description="Decision Tree vs Random Forest on Titanic with metrics")
    parser.add_argument("--csv", type=str, default=None, help="Path to titanic.csv")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    args = parser.parse_args()

    df = load_data(args.csv)
    X, y = split_xy(df)
    preprocessor, used_features = build_preprocessor(df)

    print("Using features:", used_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Models
    dt = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ])

    rf = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])

    # Train
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predict
    dt_pred = dt.predict(X_test)
    rf_pred = rf.predict(X_test)

    # Probabilities (for ROC AUC)
    def proba_or_none(pipe):
        try:
            return pipe.predict_proba(X_test)[:, 1]
        except Exception:
            return None

    dt_proba = proba_or_none(dt)
    rf_proba = proba_or_none(rf)

    # Evaluate
    evaluate("Decision Tree", y_test, dt_pred, dt_proba)
    evaluate("Random Forest", y_test, rf_pred, rf_proba)

    # Quick reflection prompts (prints for your notes)
    print("\nReflection:")
    print("- Which model has higher recall (caught more actual survivors)?")
    print("- Which has higher precision (fewer false alarms)?")
    print("- If you had to prioritize saving lives, which metric would you optimize and why?")

if __name__ == "__main__":
    main()
