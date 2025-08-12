# AI/simulations/planes/scripts/generate_full_report.py
"""
Build a SINGLE self-contained HTML report that includes:
  • One-Page Summary for Plane 1, 2, 3, 4
  • One-Page Summary combining all 4 planes
  • Plane 5 Predicted Loss Report (projected from planes 1–4; no manifest)

All charts are embedded as base64 images so the HTML is standalone.

Run from repo root (PowerShell):
  python -m pip install -U numpy pandas matplotlib
  python AI\simulations\planes\scripts\generate_full_report.py
"""

from __future__ import annotations
import io, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from textwrap import dedent

# ------------------------- Paths -------------------------
PLANES_DIR  = Path(__file__).resolve().parents[1]      # .../planes
DATA_DIR    = PLANES_DIR / "data"
REPORTS_DIR = PLANES_DIR / "reports"
for p in (DATA_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

OUT_HTML = REPORTS_DIR / "planes_all_in_one.html"

# ------------------------- Data gen -------------------------
def _clip(a, lo, hi):
    return np.minimum(np.maximum(a, lo), hi)

def gen_plane_manifest(seed: int, n: int = 50, with_labels: bool = True) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    sex  = r.choice(["male", "female"], size=n, p=[0.6, 0.4])
    age  = _clip(r.normal(35, 15, size=n), 0, 80).round(1)
    clas = r.choice([1, 2, 3], size=n, p=[0.25, 0.35, 0.40])
    seat_row  = r.integers(1, 31, size=n)
    seat_zone = np.where(seat_row <= 10, "Front",
                  np.where(seat_row <= 20, "Middle", "Rear"))
    near_exit = np.isin(seat_row, [1,2,10,11,20,21,29,30]).astype(int)
    seat_pos  = r.choice(["Window","Middle","Aisle"], size=n, p=[0.4,0.2,0.4])
    aisle     = (seat_pos == "Aisle").astype(int)
    fare = np.select(
        [clas==1, clas==2, clas==3],
        [r.uniform(250,500,size=n), r.uniform(150,300,size=n), r.uniform(60,160,size=n)]
    ).round(2)
    gate = r.choice(["A","B","C"], size=n, p=[0.5,0.3,0.2])
    luggage = r.choice([0,1,2,3], size=n, p=[0.15,0.55,0.25,0.05])
    with_children = ((age < 40) & (r.random(size=n) < 0.25)).astype(int)

    df = pd.DataFrame({
        "PassengerId": np.arange(1, n+1),
        "Sex": sex, "Age": age, "Class": clas,
        "SeatRow": seat_row, "SeatZone": seat_zone, "NearExit": near_exit,
        "SeatPos": seat_pos, "AisleSeat": aisle,
        "Fare": fare, "Gate": gate, "Luggage": luggage, "WithChildren": with_children
    })

    if with_labels:
        logit = -0.5
        logit += 1.0 * (df["Sex"] == "female").astype(int)
        logit += 0.9 * (df["Class"] == 1).astype(int) + 0.3 * (df["Class"] == 2).astype(int)
        logit += 0.5 * (df["Age"] < 12).astype(int)
        logit -= 0.6 * (df["Age"] >= 65).astype(int)
        logit += -0.01 * _clip(df["Age"] - 45, 0, 100)
        logit += 0.6  * df["NearExit"]
        logit += 0.25 * df["AisleSeat"]
        logit += 0.15 * (df["SeatZone"] == "Front").astype(int) - 0.15 * (df["SeatZone"] == "Rear").astype(int)
        logit += -0.25 * (df["Luggage"] >= 2).astype(int)
        logit += np.random.default_rng(seed+999).normal(0, 0.35, size=n)

        p = 1 / (1 + np.exp(-logit))
        survived = (np.random.default_rng(seed+123).random(size=n) < p).astype(int)
        df["Survived"] = survived
        df["SurvivalProb_true"] = p.round(3)

    return df

# ------------------------- Utils -------------------------
def rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = (df.groupby(by)["Survived"]
           .agg(n="size", survived="sum", rate="mean")
           .assign(rate_pct=lambda d: (d["rate"]*100).round(1)))
    try:
        return g.sort_index()
    except Exception:
        return g

def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"

def bar_png(tbl: pd.DataFrame, title: str, ylabel: str) -> str:
    fig = plt.figure(figsize=(6.6, 4.6))
    ax = fig.add_subplot(111)
    labels = list(tbl.index.astype(str))
    ax.bar(range(len(labels)), tbl["rate"])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel); ax.set_title(title)
    for i, (rate, n) in enumerate(zip(tbl["rate"], tbl["n"])):
        ax.text(i, min(rate+0.02, 0.96), f"{rate*100:.1f}%\n(n={n})", ha="center", va="bottom")
    return fig_to_data_uri(fig)

def section_plane(df: pd.DataFrame, plane_title: str) -> str:
    overall = df["Survived"].mean()
    sex_tbl   = rate_table(df, "Sex")
    class_tbl = rate_table(df, "Class")
    zone_tbl  = rate_table(df, "SeatZone").reindex(["Front","Middle","Rear"])

    img_sex   = bar_png(sex_tbl,   f"{plane_title}: Survival by Sex", "Survival rate")
    img_class = bar_png(class_tbl, f"{plane_title}: Survival by Class", "Survival rate")
    img_zone  = bar_png(zone_tbl,  f"{plane_title}: Survival by Seat Zone", "Survival rate")

    html = f"""\
<section>
  <h2>{plane_title} — One-Page Summary</h2>
  <p class="muted">Passengers: {len(df)} • Overall survival: {overall*100:.1f}%</p>
  <div class="grid">
    <div class="card"><h3>By Sex</h3><img src="{img_sex}" /></div>
    <div class="card"><h3>By Class</h3><img src="{img_class}" /></div>
    <div class="card"><h3>By Seat Zone</h3><img src="{img_zone}" /></div>
  </div>
</section>
"""
    return html

def section_overall(all_df: pd.DataFrame) -> str:
    overall = all_df["Survived"].mean()
    sex_tbl   = rate_table(all_df, "Sex")
    class_tbl = rate_table(all_df, "Class")
    zone_tbl  = rate_table(all_df, "SeatZone").reindex(["Front","Middle","Rear"])

    img_sex   = bar_png(sex_tbl,   "All Planes: Survival by Sex", "Survival rate")
    img_class = bar_png(class_tbl, "All Planes: Survival by Class", "Survival rate")
    img_zone  = bar_png(zone_tbl,  "All Planes: Survival by Seat Zone", "Survival rate")

    html = f"""\
<section>
  <h2>All Four Planes — One-Page Summary</h2>
  <p class="muted">Passengers: {len(all_df)} • Overall survival: {overall*100:.1f}%</p>
  <div class="grid">
    <div class="card"><h3>By Sex</h3><img src="{img_sex}" /></div>
    <div class="card"><h3>By Class</h3><img src="{img_class}" /></div>
    <div class="card"><h3>By Seat Zone</h3><img src="{img_zone}" /></div>
  </div>
</section>
"""
    return html

def section_plane5_projection(all_df: pd.DataFrame, n_plane5: int = 50) -> str:
    # composition & rates by (Sex,Class,SeatZone)
    key = ["Sex","Class","SeatZone"]
    comp = (all_df.groupby(key).size().rename("count").reset_index())
    total = comp["count"].sum()
    comp["weight"] = comp["count"] / total
    rates = (all_df.groupby(key)["Survived"].mean().rename("rate").reset_index())
    pooled = comp.merge(rates, on=key, how="left")

    exp_survivors   = (n_plane5 * (pooled["weight"] * pooled["rate"])).sum()
    exp_casualties  = n_plane5 - exp_survivors
    p_overall       = all_df["Survived"].mean()
    se              = (p_overall * (1 - p_overall) / n_plane5) ** 0.5
    ci_low, ci_high = max(0.0, p_overall - 1.96 * se), min(1.0, p_overall + 1.96 * se)

    # marginals for charts (projected)
    def marginal(by):
        g = (all_df.groupby(by)["Survived"]
             .agg(n="size", survived="sum", rate="mean"))
        return g.sort_index()

    img_sex   = bar_png(marginal("Sex"),   "Plane 5 (Projected): Survival by Sex", "Predicted survival rate")
    img_class = bar_png(marginal("Class"), "Plane 5 (Projected): Survival by Class", "Predicted survival rate")
    img_zone  = bar_png(marginal("SeatZone").reindex(["Front","Middle","Rear"]),
                        "Plane 5 (Projected): Survival by Seat Zone", "Predicted survival rate")

    html = f"""\
<section>
  <h2>Plane 5 — Predicted Loss Report (No Manifest)</h2>
  <p class="muted">
    Assumes Plane 5 has the <b>same passenger mix</b> (Sex × Class × SeatZone) as planes 1–4.<br>
    Passengers: {n_plane5} • Expected survivors: {exp_survivors:.1f} • Expected casualties: {exp_casualties:.1f}<br>
    Expected survival rate: {(exp_survivors/n_plane5)*100:.1f}% (95% CI ~ {ci_low*100:.1f}% – {ci_high*100:.1f}%)
  </p>
  <div class="grid">
    <div class="card"><h3>By Sex</h3><img src="{img_sex}" /></div>
    <div class="card"><h3>By Class</h3><img src="{img_class}" /></div>
    <div class="card"><h3>By Seat Zone</h3><img src="{img_zone}" /></div>
  </div>
  <p class="muted">Method: pooled rates r(s,c,z) and composition weights w(s,c,z) from planes 1–4.
  Expected survivors = N × Σ w_g × r_g.</p>
</section>
"""
    return html

# ------------------------- Main -------------------------
def main(n_per_plane: int = 50, include_plane5: bool = True):
    # Generate planes 1–4 (with labels)
    seeds = [101, 202, 303, 404]
    planes = []
    sections = []

    for i, seed in enumerate(seeds, start=1):
        df = gen_plane_manifest(seed, n=n_per_plane, with_labels=True)
        planes.append(df)
        # Save CSV for reference (not needed by HTML)
        (DATA_DIR / f"plane{i}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        sections.append(section_plane(df, f"Plane {i}"))

    all_df = pd.concat(planes, ignore_index=True)
    sections.insert(0, section_overall(all_df))  # put combined summary first

    if include_plane5:
        sections.append(section_plane5_projection(all_df, n_plane5=n_per_plane))

    # Assemble final HTML
    html = f"""\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Planes — All-in-One Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; line-height: 1.55; }}
h1 {{ margin: 0 0 12px; }}
h2 {{ margin: 22px 0 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
.card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; }}
.muted {{ color: #6b7280; }}
hr {{ border: 0; border-top: 1px solid #eee; margin: 18px 0; }}
img {{ max-width: 100%; height: auto; border-radius: 8px; }}
.nav a {{ margin-right: 12px; }}
</style>
</head>
<body>
  <h1>Planes — One-Page Summaries (1 file)</h1>
  <p class="muted">This report is self-contained. Charts are embedded as images inside the HTML.</p>
  <div class="nav">
    <a href="#all">All 4 Planes</a>
    <a href="#p1">Plane 1</a>
    <a href="#p2">Plane 2</a>
    <a href="#p3">Plane 3</a>
    <a href="#p4">Plane 4</a>
    {"<a href=\"#p5\">Plane 5 (Projected)</a>" if include_plane5 else ""}
  </div>
  <hr/>

  <a id="all"></a>
  {sections[0]}
  <hr/>

  <a id="p1"></a>
  {sections[1]}
  <hr/>

  <a id="p2"></a>
  {sections[2]}
  <hr/>

  <a id="p3"></a>
  {sections[3]}
  <hr/>

  <a id="p4"></a>
  {sections[4]}
  <hr/>

  {"<a id=\"p5\"></a>" + sections[5] if include_plane5 else ""}
</body>
</html>
"""
    OUT_HTML.write_text(dedent(html), encoding="utf-8")
    print(f"[OK] Wrote {OUT_HTML} (open in your browser).")

if __name__ == "__main__":
    main()