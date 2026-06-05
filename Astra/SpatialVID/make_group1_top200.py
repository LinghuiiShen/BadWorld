import pandas as pd

csv_path = "/root/autodl-tmp/cache/hub/datasets--SpatialVID--SpatialVID-HQ/snapshots/97e22f97447c012ef7460e5989b83a674643a880/data/train/SpatialVID_HQ_metadata.csv"

df = pd.read_csv(csv_path, skiprows=1, low_memory=False)
df.columns = df.columns.str.strip()

df["group id"] = pd.to_numeric(df["group id"], errors="coerce")

sub = df[df["group id"] == 1].copy().head(150)

cols = [
    "id", "group id", "video path", "annotation path",
    "sceneType", "fps", "num frames", "aesthetic score",
    "motion score", "timeOfDay", "weather", "crowdDensity"
]
cols = [c for c in cols if c in sub.columns]

sub[cols].to_csv("group1_top150.csv", index=False)

print(f"saved {len(sub)} rows to group1_top150.csv")
print(sub[cols].head(10).to_string(index=False))