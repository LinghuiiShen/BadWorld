"""
Video information extraction utility supporting multiple backends (OpenCV, TorchVision, AV).
"""

import argparse
import os
import random
import cv2
import av
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures


def get_video_length(cap, method="header"):
    """Get video frame count using different methods"""
    assert method in ["header", "set"]
    if method == "header":
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        length = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    return length


def get_video_info(args):
    """Extract video information using specified backend"""
    idx, path, backend = args
    try:
        if backend == "torchvision":
            from tools.utils.read_video import read_video

            vframes, infos = read_video(path)
            num_frames, height, width = (
                vframes.shape[0],
                vframes.shape[2],
                vframes.shape[3],
            )
            fps = (
                float(infos.get("video_fps", np.nan))
                if isinstance(infos, dict)
                else np.nan
            )
        elif backend == "opencv":
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError("Video open failed")
            num_frames = get_video_length(cap, method="header")
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
        elif backend == "av":
            container = av.open(path)
            stream = container.streams.video[0]
            num_frames = int(stream.frames)
            height = int(stream.height)
            width = int(stream.width)
            if stream.average_rate is not None:
                fps = float(stream.average_rate)
            elif stream.guessed_rate is not None:
                fps = float(stream.guessed_rate)
            else:
                fps = np.nan
        else:
            raise ValueError("Unknown backend")

        # Calculate derived metrics
        hw = height * width
        aspect_ratio = height / width if width > 0 else np.nan
        return (idx, True, num_frames, height, width, aspect_ratio, hw, fps)
    except Exception:
        return (idx, False, 0, 0, 0, np.nan, np.nan, np.nan)


def main(args):
    """Main function to extract video information"""
    # Load data
    data = pd.read_csv(args.csv_path)
    if data.empty:
        data.to_csv(args.csv_save_path, index=False)
        print(f"Input CSV is empty. Saved 0 samples to {args.csv_save_path}.")
        return

    tasks = [(index, row["video_path"], args.backend) for index, row in data.iterrows()]
    num_workers = args.num_workers if args.num_workers else os.cpu_count() or 1

    # Process videos with a per-video progress bar (more intuitive than per-worker)
    if args.disable_parallel or num_workers <= 1:
        ret = [
            get_video_info(task)
            for task in tqdm(tasks, total=len(tasks), desc="Processing videos")
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            ret = list(
                tqdm(
                    executor.map(get_video_info, tasks, chunksize=16),
                    total=len(tasks),
                    desc="Processing videos",
                )
            )

    ret.sort(key=lambda x: x[0])
    (
        _idx_list,
        success_list,
        num_frames_list,
        height_list,
        width_list,
        aspect_ratio_list,
        hw_list,
        fps_list,
    ) = zip(*ret)

    # Add extracted information to DataFrame
    data["success"] = success_list
    data["num_frames"] = num_frames_list
    data["height"] = height_list
    data["width"] = width_list
    data["aspect_ratio"] = aspect_ratio_list
    data["resolution"] = hw_list
    data["fps"] = fps_list

    # Filter existing files if requested
    if args.ext:
        assert "video_path" in data.columns
        data = data[data["video_path"].apply(os.path.exists)]

    # Sort by frame count
    if "num_frames" in data.columns:
        data = data.sort_values(by="num_frames", ascending=True)

    data.to_csv(args.csv_save_path, index=False)
    print(f"Saved {len(data)} samples to {args.csv_save_path}.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract video information using multiple backends"
    )
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--csv_save_path", type=str, default=None, help="Path to save output CSV file"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="opencv",
        help="Video backend",
        choices=["opencv", "torchvision", "av"],
    )
    parser.add_argument(
        "--disable-parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Number of parallel workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # File existence checking
    parser.add_argument("--ext", action="store_true", help="Check if video files exist")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    main(args)
