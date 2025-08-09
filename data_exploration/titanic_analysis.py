import pandas as pd
from pathlib import Path

# ---------- Setup ----------
DATA_PATH = Path(__file__).with_name("titanic.csv")
assert DATA_PATH.exists(), f"CSV not found at {DATA_PATH}"

pd.set_option("display.max_rows", 100)

# ---------- Load ----------
df = pd.read_csv(DATA_PATH)

# Quick overall baseline
overall = df["Survived"].mean()
print(f"\nOverall survival rate: {overall:.3f} ({overall*100:.1f}%)")

# ---------- Helper to compute counts + rates ----------
def rates(data, by):
    out = (
        data.groupby(by)["Survived"]
        .agg(n="size", survived="sum", rate="mean")
        .assign(rate_pct=lambda d: (d["rate"] * 100).round(1))
        .sort_values("rate", ascending=False)
    )
    return out

# 1) Gender Impact
print("\n# 1) Gender Impact — survival by Sex")
print(rates(df, "Sex"))

# 2) Class Impact
print("\n# 2) Class Impact — survival by Pclass")
print(rates(df, "Pclass").sort_index())  # keep 1,2,3 order

# 3) Age Factor — bin ages into groups
bins = [0, 12, 18, 35, 50, 80]
labels = ["Child", "Teen", "Young Adult", "Adult", "Senior"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True)
print("\n# 3) Age Factor — survival by AgeGroup")
print(rates(df.dropna(subset=["AgeGroup"]), "AgeGroup").reindex(labels))

# 4) Fare Influence — terciles (Low/Medium/High)
df["FareGroup"] = pd.qcut(df["Fare"], q=3, labels=["Low", "Medium", "High"])
print("\n# 4) Fare Influence — survival by FareGroup")
print(rates(df, "FareGroup").reindex(["Low", "Medium", "High"]))

# 5) Port Impact — include missing as 'Unknown'
df["Embarked"] = df["Embarked"].fillna("Unknown")
print("\n# 5) Port Impact — survival by Embarked")
print(rates(df, "Embarked"))
