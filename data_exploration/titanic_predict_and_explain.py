import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Predict Titanic survival for one passenger and explain why (simple or engineer mode)."
    )
    ap.add_argument("--passenger", type=int, required=True,
                    help="Passenger number. Use --by row (1..N) or --by id (PassengerId value).")
    ap.add_argument("--by", choices=["row", "id"], default="row",
                    help="Interpret --passenger as dataframe row (1-based) or PassengerId.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for class label (default 0.5).")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test size fraction (default 0.2).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--mode", choices=["simple", "engineer"], default="simple",
                    help="Output style: simple (human) or engineer (detailed).")
    ap.add_argument("--top", type=int, default=2,
                    help="How many top positive/negative reasons to show in simple mode.")
    return ap.parse_args()

# ---------------- Data ----------------
def load_data():
    data_path = Path(__file__).with_name("titanic.csv")
    assert data_path.exists(), f"CSV not found at {data_path}"
    df = pd.read_csv(data_path)

    features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
    target = "Survived"
    missing = [c for c in [*features, target] if c not in df.columns]
    assert not missing, f"Missing columns in CSV: {missing}"

    X = df[features].copy()
    y = df[target].astype(int)
    pid_col = "PassengerId" if "PassengerId" in df.columns else None
    return df, X, y, features, pid_col

# ---------------- Preprocess ----------------
def build_preprocess():
    num_features = ["Age", "Fare", "Pclass"]
    cat_features = ["Sex", "Embarked"]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features)
    ])
    return preprocess, num_features, cat_features

# ---------------- Train ----------------
def train_pipeline(X, y, preprocess, seed, test_size, mode):
    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    pipe.fit(X_tr, y_tr)

    # Show metrics only in engineer mode
    if mode == "engineer":
        y_pred = pipe.predict(X_te)
        y_proba = pipe.predict_proba(X_te)[:, 1]
        print("\n=== Test Metrics ===")
        print(f"Accuracy : {accuracy_score(y_te, y_pred):.3f}")
        print(f"Precision: {precision_score(y_te, y_pred, zero_division=0):.3f}")
        print(f"Recall   : {recall_score(y_te, y_pred, zero_division=0):.3f}")
        print(f"F1-score : {f1_score(y_te, y_pred, zero_division=0):.3f}")
        print(f"ROC-AUC  : {roc_auc_score(y_te, y_proba):.3f}")
        print("\n=== Classification Report ===")
        print(classification_report(y_te, y_pred, digits=3, zero_division=0))

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        print("\n=== 5-fold CV ===")
        print(f"Accuracy: mean {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
        print(f"ROC-AUC : mean {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    return pipe

# ---------------- Names after transform ----------------
def get_transformed_feature_names(pipe, cat_features, num_features):
    ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(cat_features))
    return list(num_features) + cat_names

# ---------------- Pretty names ----------------
def pretty_feature(fname, raw_row):
    # Map technical names to human phrases
    if fname == "Age":
        return f"Age ({raw_row['Age'] if pd.notna(raw_row['Age']) else 'imputed'})"
    if fname == "Fare":
        return f"Fare paid ({raw_row['Fare'] if pd.notna(raw_row['Fare']) else 'imputed'})"
    if fname == "Pclass":
        return f"Passenger class ({int(raw_row['Pclass'])})"
    if fname.startswith("Sex_"):
        return "Being female" if fname.endswith("female") else "Being male"
    if fname.startswith("Embarked_"):
        code = fname.split("_", 1)[1]
        port = {"S":"Southampton", "C":"Cherbourg", "Q":"Queenstown"}.get(code, code)
        return f"Boarded at {port}"
    return fname

# ---------------- Explain one prediction ----------------
def explain_single(pipe, X, df_full, by, passenger_num, pid_col, threshold, mode, topk):
    # locate row
    if by == "id":
        assert pid_col is not None, "PassengerId column not found; use --by row instead."
        assert passenger_num in set(df_full[pid_col]), f"PassengerId {passenger_num} not in data."
        row = df_full.index[df_full[pid_col] == passenger_num][0]
    else:
        assert 1 <= passenger_num <= len(X), f"Row must be in 1..{len(X)}"
        row = passenger_num - 1

    x_raw = X.iloc[[row]]
    proba = pipe.predict_proba(x_raw)[:, 1][0]
    pred = int(proba >= threshold)

    clf = pipe.named_steps["clf"]
    prep = pipe.named_steps["prep"]
    X_trans = prep.transform(x_raw)               # shape (1, n_features)
    coef = clf.coef_.ravel()
    intercept = clf.intercept_[0]

    num_features = ["Age", "Fare", "Pclass"]
    cat_features = ["Sex", "Embarked"]
    feat_names = get_transformed_feature_names(pipe, cat_features, num_features)

    values = np.asarray(X_trans).ravel()
    contrib = values * coef
    contrib_df = pd.DataFrame({
        "feature": feat_names,
        "value": values,
        "coef": coef,
        "contribution": contrib
    })

    # Keep only active one-hot features + all numeric
    active = (contrib_df["value"] != 0) | (contrib_df["feature"].isin(num_features))
    contrib_df = contrib_df[active].copy()

    # Sort
    contrib_df.sort_values("contribution", ascending=False, inplace=True)

    # Humanized labels
    raw_row = X.iloc[row]
    contrib_df["label"] = contrib_df["feature"].apply(lambda n: pretty_feature(n, raw_row))

    # SIMPLE MODE OUTPUT
    if mode == "simple":
        print(f"\nPassenger {passenger_num} — Prediction Report")
        print("-" * 40)
        print(f"Prediction: {'Survived' if pred==1 else 'Did not survive'} ({proba*100:.0f}% chance)")

        top_pos = contrib_df.head(topk)
        top_neg = contrib_df.tail(topk).sort_values("contribution")

        def to_sentence(series):
            # positive = helped; negative = hurt
            sents = []
            for _, r in series.iterrows():
                verb = "increased" if r["contribution"] > 0 else "decreased"
                sents.append(f"- {r['label']} {verb} the chance")
            return "\n".join(sents) if len(sents) else "- (no strong factors)"

        print("\nMain reasons:")
        print(to_sentence(top_pos))
        print(to_sentence(top_neg))

        if "Survived" in df_full.columns:
            actual = int(df_full.iloc[row]["Survived"])
            print(f"\nActual outcome in data: {'Survived' if actual==1 else 'Did not survive'}")

        print("\n(We used a simple logistic model. Positive factors push the chance up; negative factors push it down.)")
        return

    # ENGINEER MODE OUTPUT
    print("\n=== Passenger input (raw) ===")
    cols = ["Pclass","Sex","Age","Fare","Embarked"]
    print(x_raw[cols].to_string(index=False))

    print("\n=== Prediction ===")
    print(f"Probability of survival: {proba:.3f}")
    print(f"Predicted class @ threshold {threshold:.2f}: {'SURVIVED (1)' if pred==1 else 'NOT SURVIVED (0)'}")
    if "Survived" in df_full.columns:
        print(f"(Actual label in dataset: {int(df_full.iloc[row]['Survived'])})")

    top_for = contrib_df.head(5)[["label","contribution"]]
    top_against = contrib_df.tail(5)[["label","contribution"]].sort_values("contribution")

    print("\n=== Top reasons pushing TOWARD survive (positive contributions) ===")
    for _, r in top_for.iterrows():
        print(f"{r['label']:>28s} : +{r['contribution']:.3f}")

    print("\n=== Top reasons pushing AGAINST survive (negative contributions) ===")
    for _, r in top_against.iterrows():
        print(f"{r['label']:>28s} : {r['contribution']:.3f}")

    up = top_for.iloc[0]
    down = top_against.iloc[0]
    print("\n=== Short explanation ===")
    print(f"We predicted {'SURVIVED' if pred==1 else 'NOT SURVIVED'} mainly because "
          f"{up['label']} contributed most positively, while {down['label']} "
          f"contributed most negatively. (Contributions are in log-odds space.)")

# ---------------- Main ----------------
def main():
    args = parse_args()
    df, X, y, features, pid_col = load_data()
    preprocess, num_features, cat_features = build_preprocess()
    pipe = train_pipeline(X, y, preprocess, seed=args.seed, test_size=args.test_size, mode=args.mode)
    explain_single(pipe, X, df, by=args.by, passenger_num=args.passenger,
                   pid_col=pid_col, threshold=args.threshold, mode=args.mode, topk=args.top)

if __name__ == "__main__":
    main()