#!/usr/bin/env python3
"""
One-shot HTML report:
- Ingest raw per-match table (player_match_table.csv)
- Build rolling features + "plays_next" label
- Train LogisticRegression, DecisionTree, RandomForest (time-aware split)
- Pick best (macro F1) and score latest row per player
- Emit a single styled HTML with: summary, metrics, top N, full table, and RF importances (if RF wins)

Usage:
  python make_report.html.py \
    --input ../data/player_match_table.csv \
    --output ../reports/players_next_report.html \
    --threshold 0.50 \
    --top_n 25
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -------------------------------
# Feature engineering (inline)
# -------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["player_id", "match_date"])

    # label: did the player play minutes in their NEXT match?
    df["minutes_next"] = df.groupby("player_id")["minutes"].shift(-1)
    df["plays_next"] = (df["minutes_next"].fillna(0) > 0).astype(int)

    def _per_player(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["minutes_last_1"] = g["minutes"].shift(1).fillna(0)
        g["minutes_last_3"] = (
            g["minutes"].shift(1).rolling(3, min_periods=1).sum().fillna(0)
        )
        g["started_last_match"] = g["started"].shift(1).fillna(0).astype(int)
        prev_date = g["match_date"].shift(1)
        g["days_since_last_appearance"] = (g["match_date"] - prev_date).dt.days.fillna(30)

        # prior matches in last 14 days (strictly before current)
        dates = g["match_date"].values
        counts = []
        for i, d in enumerate(dates):
            mask = (dates < d) & (dates >= (d - np.timedelta64(14, "D")))
            counts.append(int(mask.sum()))
        g["matches_last_14d"] = counts

        for col in ["yellow_cards", "red_cards"]:
            if col in g.columns:
                g[f"{col}_last_5"] = (
                    g[col].shift(1).rolling(5, min_periods=1).sum().fillna(0)
                )
        return g

    df = df.groupby("player_id", group_keys=False).apply(_per_player)

    keep = [
        "player_id", "player_name", "team", "opponent", "position",
        "home_away", "match_id", "match_date",
        "minutes_last_1", "minutes_last_3", "started_last_match",
        "days_since_last_appearance", "matches_last_14d",
        "plays_next",
    ]
    if "yellow_cards_last_5" in df.columns: keep.append("yellow_cards_last_5")
    if "red_cards_last_5" in df.columns: keep.append("red_cards_last_5")
    if "injured_at_kickoff" in df.columns: keep.append("injured_at_kickoff")
    if "suspended_at_kickoff" in df.columns: keep.append("suspended_at_kickoff")

    out = df[keep].dropna(subset=["plays_next"]).reset_index(drop=True)
    return out

def latest_row_per_player(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "match_date"])
    return df.groupby("player_id").tail(1).reset_index(drop=True)

def time_aware_split(df: pd.DataFrame, date_col="match_date", frac=0.2):
    dates = pd.to_datetime(df[date_col])
    cutoff = dates.quantile(1.0 - frac)
    train_idx = dates < cutoff
    valid_idx = ~train_idx
    return train_idx, valid_idx, cutoff

# -------------------------------
# HTML helpers
# -------------------------------
STYLE = """
<style>
:root { --bg:#0b0f19; --card:#111827; --muted:#9CA3AF; --ok:#10B981; --no:#EF4444; --ink:#E5E7EB; }
* { box-sizing:border-box }
body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin:24px; background:var(--bg); color:var(--ink); }
h1,h2 { margin: 0.7rem 0 0.4rem; }
.container { max-width: 1200px; margin: 0 auto; }
.card { background:var(--card); border-radius:16px; padding:18px 16px; box-shadow:0 6px 24px rgba(0,0,0,.25); margin-bottom:18px; }
.grid { display:grid; gap:12px; grid-template-columns: repeat(4, minmax(0,1fr)); }
.kpi { padding:14px; border-radius:14px; background:rgba(255,255,255,.04); text-align:center; }
.kpi h3 { font-size:0.9rem; color:var(--muted); margin:0 0 6px; }
.kpi .v { font-size:1.2rem; font-weight:700; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:700; }
.badge.yes { background:rgba(16,185,129,.15); color:#10B981; }
.badge.no  { background:rgba(239,68,68,.18); color:#F87171; }
table { width:100%; border-collapse:collapse; }
th,td { padding:8px 10px; border-bottom:1px solid rgba(255,255,255,.08); text-align:left; }
th { color:var(--muted); cursor:pointer; position:sticky; top:0; background:var(--card); }
tr:hover { background:rgba(255,255,255,.04); }
pre { white-space:pre-wrap; color:#E5E7EB; background:rgba(255,255,255,.04); padding:12px; border-radius:12px; }
a { color:#93C5FD; text-decoration:none; }
.footer { color:var(--muted); font-size:0.9rem; }
</style>
<script>
function sortTable(id, col){
  const t = document.getElementById(id); const tb = t.tBodies[0];
  const asc = t.getAttribute('data-asc') !== 'true';
  const rows = Array.from(tb.rows);
  rows.sort((a,b)=>{
    const A=a.cells[col].innerText.trim(), B=b.cells[col].innerText.trim();
    const nA=parseFloat(A.replace(/[^0-9.-]/g,'')), nB=parseFloat(B.replace(/[^0-9.-]/g,''));
    const bothNum=!isNaN(nA)&&!isNaN(nB);
    if(bothNum) return asc? nA-nB : nB-nA;
    return asc? A.localeCompare(B) : B.localeCompare(A);
  });
  rows.forEach(r=>tb.appendChild(r)); t.setAttribute('data-asc', asc?'true':'false');
}
function enableSort(id){ document.querySelectorAll('#'+id+' th').forEach((th,i)=> th.onclick=()=>sortTable(id,i)); }
</script>
"""

def df_to_html(df: pd.DataFrame, table_id: str) -> str:
    # pandas.DataFrame.to_html renders DataFrames as HTML tables; we enable escaping=False
    # so we can show the colored prediction badges. (pandas docs)
    return df.to_html(index=False, escape=False, table_id=table_id)

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/player_match_table.csv")
    ap.add_argument("--output", default="../reports/players_next_report.html")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--top_n", type=int, default=25)
    args = ap.parse_args()

    raw = pd.read_csv(args.input)
    feats = compute_features(raw)
    feats["match_date"] = pd.to_datetime(feats["match_date"])

    # features
    num_features = [
        "minutes_last_1", "minutes_last_3", "started_last_match",
        "days_since_last_appearance", "matches_last_14d",
    ]
    for c in ["yellow_cards_last_5", "red_cards_last_5", "injured_at_kickoff", "suspended_at_kickoff"]:
        if c in feats.columns: num_features.append(c)

    cat_features = [c for c in ["home_away", "position"] if c in feats.columns]

    # drop incomplete rows for training
    target = "plays_next"
    required = num_features + cat_features + [target, "match_date"]
    data = feats.dropna(subset=required).reset_index(drop=True)

    # time-aware split (no leakage)
    train_idx, valid_idx, cutoff = time_aware_split(data, "match_date", frac=0.2)
    X_train = data.loc[train_idx, num_features + cat_features]
    y_train = data.loc[train_idx, target].astype(int)
    X_valid = data.loc[valid_idx, num_features + cat_features]
    y_valid = data.loc[valid_idx, target].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(random_state=0, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, random_state=0, n_jobs=-1, class_weight="balanced_subsample"
        ),
    }

    reports = []
    best = {"name": None, "f1": -1.0, "pipe": None}

    for name, est in models.items():
        pipe = Pipeline([("pre", pre), ("clf", est)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_valid)
        cr = classification_report(y_valid, preds, digits=3)
        reports.append((name, cr))
        try:
            macro_line = [ln for ln in cr.splitlines() if "macro avg" in ln][0]
            macro_f1 = float(macro_line.split()[-2])
        except Exception:
            macro_f1 = -1.0
        if macro_f1 > best["f1"]:
            best.update(name=name, f1=macro_f1, pipe=pipe)

    # score latest row per player
    latest = latest_row_per_player(feats)
    X_score = latest[num_features + cat_features]
    proba = best["pipe"].predict_proba(X_score)[:, 1] if hasattr(best["pipe"], "predict_proba") else None
    if proba is None:
        # logistic fallback from decision_function (rarely needed here)
        s = best["pipe"].decision_function(X_score); proba = 1/(1+np.exp(-s))

    latest = latest.copy()
    latest["pred_proba_play"] = proba
    latest["pred_class"] = (latest["pred_proba_play"] >= args.threshold).astype(int)
    latest["will_play"] = np.where(latest["pred_class"]==1, '<span class="badge yes">YES</span>',
                                                     '<span class="badge no">NO</span>')
    display_cols = [
        "player_id","player_name","team","opponent","position","home_away","match_date",
        "minutes_last_1","minutes_last_3","started_last_match",
        "days_since_last_appearance","matches_last_14d","pred_proba_play","will_play"
    ]
    display_cols = [c for c in display_cols if c in latest.columns]

    top = latest.sort_values("pred_proba_play", ascending=False).head(args.top_n)
    top_disp = top.assign(pred_proba_play=lambda d: d["pred_proba_play"].round(3))[display_cols]
    all_disp = latest.assign(pred_proba_play=lambda d: d["pred_proba_play"].round(3))[display_cols]\
                     .sort_values("pred_proba_play", ascending=False)

    # optional: feature importances for RandomForest
    imps_html = ""
    if best["name"] == "RandomForest":
        preproc = best["pipe"].named_steps["pre"]
        clf = best["pipe"].named_steps["clf"]
        num_names = preproc.named_transformers_["num"].get_feature_names_out(num_features)
        cat_names = preproc.named_transformers_["cat"].get_feature_names_out(cat_features) if cat_features else np.array([])
        names = np.concatenate([num_names, cat_names])
        imps = pd.DataFrame({"feature": names, "importance": clf.feature_importances_})\
                .sort_values("importance", ascending=False).head(30)
        imps_html = df_to_html(imps, "tbl_imps")

    # build HTML
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary = f"""
    <div class="card">
      <h1>Players — Next Match Prediction</h1>
      <div class="grid" style="grid-template-columns:repeat(4,minmax(0,1fr)); margin-top:8px;">
        <div class="kpi"><h3>Best model</h3><div class="v">{best['name']}</div></div>
        <div class="kpi"><h3>Macro F1 (valid)</h3><div class="v">{best['f1']:.3f}</div></div>
        <div class="kpi"><h3>Cutoff date (80/20)</h3><div class="v">{pd.to_datetime(data['match_date']).quantile(0.8).date()}</div></div>
        <div class="kpi"><h3>Players scored</h3><div class="v">{len(latest):,}</div></div>
      </div>
      <div style="color:var(--muted); margin-top:6px;">Generated {now} — threshold {args.threshold:.2f} (predict “YES” if prob ≥ threshold)</div>
    </div>
    """

    metrics_html = "\n".join(
        f"<h3 style='margin:0.2rem 0 0.4rem'>{name}</h3><pre>{cr}</pre>" for name, cr in reports
    )
    top_html = df_to_html(top_disp, "tbl_top")
    all_html = df_to_html(all_disp, "tbl_all")

    sources = """
    <div class="card footer">
      <div><strong>Sources</strong></div>
      <ul>
        <li>Pandas: <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_html.html">DataFrame.to_html</a></li>
        <li>scikit-learn Pipeline: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html">docs</a>; ColumnTransformer: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html">docs</a></li>
        <li>Estimators — LogisticRegression, DecisionTreeClassifier, RandomForestClassifier: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">LR</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">DT</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">RF</a></li>
        <li>Classification metrics text: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html">classification_report</a></li>
        <li>Time-aware CV guidance: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html">TimeSeriesSplit</a></li>
      </ul>
    </div>
    """

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Players Next Match — Report</title>
{STYLE}
</head>
<body>
  <div class="container">
    {summary}

    <div class="card">
      <h2>Top predicted to play</h2>
      {top_html}
      <script>enableSort('tbl_top');</script>
    </div>

    <div class="card">
      <h2>All predictions</h2>
      {all_html}
      <script>enableSort('tbl_all');</script>
    </div>

    <div class="card">
      <h2>Training metrics</h2>
      {metrics_html}
    </div>

    {('<div class="card"><h2>Feature importances (top 30)</h2>'+imps_html+'</div>') if imps_html else ''}

    {sources}
  </div>
</body></html>"""

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"[make_report] wrote: {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
