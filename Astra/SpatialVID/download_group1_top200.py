import os
import pandas as pd
from huggingface_hub import hf_hub_download

repo_id = "SpatialVID/SpatialVID-HQ"
repo_type = "dataset"

csv_path = "group1_top150.csv"
out_dir = "./SpatialVID_group1_top150"

os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)

for i, row in df.iterrows():
    video_path = row["video path"]
    ann_path = str(row["annotation path"]).rstrip("/")

    print(f"[{i+1}/{len(df)}] downloading video: {video_path}")
    hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=video_path,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )

    for name in ["caption.json", "poses.npy"]:
        file_path = f"{ann_path}/{name}"
        print(f"[{i+1}/{len(df)}] downloading ann: {file_path}")
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=file_path,
            local_dir=out_dir,
            local_dir_use_symlinks=False,
        )

print("done")