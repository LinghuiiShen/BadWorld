"""
High-speed video cutting utility using FFmpeg stream copy.

Features:
- No re-encoding: uses `-c copy`
- Video only: drops audio with `-map 0:v:0`
- Group tasks by source video_path for better efficiency
- Parallel processing by video group
- Per-clip progress bar
- Save successful and failed CSVs

Notes:
- This method is very fast, but not always frame-accurate.
- Clip boundaries may align to nearby keyframes depending on source encoding.
"""

import argparse
import os
import queue
import subprocess
import concurrent.futures
from multiprocessing import Manager

import pandas as pd
from scenedetect import FrameTimecode
from tqdm import tqdm


FFMPEG_PATH = "/usr/local/bin/ffmpeg"


def process_single_row(row, save_dir):
    """
    Cut one clip from source video using ffmpeg stream copy.

    Returns:
        (row_values_list, valid, error_message)
    """
    video_path = row["video_path"]
    sample_id = row["id"]
    save_path = os.path.join(save_dir, f"{sample_id}.mp4")

    # Already exists -> treat as success
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        row = row.copy()
        row["video_path"] = save_path
        return row.values.tolist(), True, ""

    if not os.path.exists(video_path):
        error_msg = f"Source video not found: {video_path} (id={sample_id})"
        return row.values.tolist(), False, error_msg

    # Parse timestamps
    try:
        fps = row["fps"]
        seg_start = FrameTimecode(timecode=row["timestamp_start"], fps=fps)
        seg_end = FrameTimecode(timecode=row["timestamp_end"], fps=fps)

        start_sec = float(seg_start.get_seconds())
        end_sec = float(seg_end.get_seconds())
        duration = end_sec - start_sec

        if duration <= 0:
            error_msg = f"Non-positive duration for id={sample_id}: {duration}"
            return row.values.tolist(), False, error_msg

    except Exception as e:
        error_msg = f"Invalid timestamp for id={sample_id}: {e}"
        return row.values.tolist(), False, error_msg

    try:
        # Fast seek + stream copy, video only, no audio.
        cmd = [
            FFMPEG_PATH,
            "-nostdin",
            "-y",
            "-ss",
            str(start_sec),
            "-t",
            str(duration),
            "-i",
            video_path,
            "-map",
            "0:v:0",
            "-an",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            save_path,
        ]

        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Verify output exists and non-empty
        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            if os.path.exists(save_path):
                os.remove(save_path)
            error_msg = f"FFmpeg produced empty/missing output for id={sample_id}"
            return row.values.tolist(), False, error_msg

        row = row.copy()
        row["video_path"] = save_path
        return row.values.tolist(), True, ""

    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
        error_msg = f"FFmpeg failed for id={sample_id}: {stderr_text}"
        if os.path.exists(save_path):
            os.remove(save_path)
        return row.values.tolist(), False, error_msg

    except Exception as e:
        error_msg = f"Unexpected error for id={sample_id}: {e}"
        if os.path.exists(save_path):
            os.remove(save_path)
        return row.values.tolist(), False, error_msg


def process_video_group(group_df, save_dir):
    """
    Process all clips from the same source video.

    Args:
        group_df: DataFrame containing rows from one source video_path
        save_dir: output clip directory

    Returns:
        list of tuples:
            (index, row_values, valid, error_msg)
    """
    results = []

    # Sort by start timestamp to make access pattern a bit more sequential
    if "timestamp_start" in group_df.columns:
        group_df = group_df.sort_values(by="timestamp_start")

    for index, row in group_df.iterrows():
        row_values, valid, error_msg = process_single_row(row, save_dir)
        results.append((index, row_values, valid, error_msg))

    return results


def worker(task_queue, results_queue, video_save_dir):
    """
    Worker that processes one video group at a time.
    """
    while True:
        try:
            video_path, group_df = task_queue.get(timeout=1)
        except queue.Empty:
            break

        try:
            group_results = process_video_group(group_df, video_save_dir)
            for item in group_results:
                results_queue.put(item)
        finally:
            task_queue.task_done()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast video cutting utility using FFmpeg stream copy"
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Input CSV path")
    parser.add_argument(
        "--csv_save_path", type=str, required=True, help="Output CSV path"
    )
    parser.add_argument(
        "--video_save_dir", type=str, required=True, help="Directory to save clips"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (defaults to CPU count)",
    )
    parser.add_argument(
        "--disable_parallel",
        action="store_true",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--drop_invalid_timestamps",
        action="store_true",
        help="Drop invalid timestamp rows and save corrected CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.csv_path):
        print(f"csv file '{args.csv_path}' not found. Exit.")
        return

    os.makedirs(args.video_save_dir, exist_ok=True)

    csv = pd.read_csv(args.csv_path)
    if len(csv) == 0:
        print("Input CSV is empty. Exit.")
        return

    required_cols = ["id", "video_path", "timestamp_start", "timestamp_end", "fps"]
    missing_cols = [c for c in required_cols if c not in csv.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    results = []

    # Group by source video
    grouped_items = list(csv.groupby("video_path", sort=False))
    total_tasks = len(csv)

    if args.disable_parallel:
        success_cnt = 0
        fail_cnt = 0

        with tqdm(total=total_tasks, desc="Processing clips", dynamic_ncols=True) as pbar:
            for video_path, group_df in grouped_items:
                group_results = process_video_group(group_df, args.video_save_dir)
                for item in group_results:
                    results.append(item)
                    _, _, valid, _ = item
                    if valid:
                        success_cnt += 1
                    else:
                        fail_cnt += 1
                    pbar.update(1)
                    pbar.set_postfix(success=success_cnt, fail=fail_cnt)

    else:
        manager = Manager()
        task_queue = manager.Queue()
        results_queue = manager.Queue()

        for video_path, group_df in grouped_items:
            task_queue.put((video_path, group_df))

        num_workers = args.num_workers if args.num_workers else os.cpu_count()
        num_workers = max(1, num_workers)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _ in range(num_workers):
                futures.append(
                    executor.submit(worker, task_queue, results_queue, args.video_save_dir)
                )

            finished = 0
            success_cnt = 0
            fail_cnt = 0

            with tqdm(total=total_tasks, desc="Processing clips", dynamic_ncols=True) as pbar:
                while finished < total_tasks:
                    try:
                        item = results_queue.get(timeout=1)
                    except queue.Empty:
                        continue

                    results.append(item)
                    finished += 1

                    _, _, valid, _ = item
                    if valid:
                        success_cnt += 1
                    else:
                        fail_cnt += 1

                    pbar.update(1)
                    pbar.set_postfix(success=success_cnt, fail=fail_cnt)

            for future in futures:
                future.result()

    # Sort back by original row index
    results.sort(key=lambda x: x[0])

    # Separate successful and failed
    success_rows = []
    failed_rows = []
    failed_errors = []

    for index, row_values, valid, error_msg in results:
        if valid:
            success_rows.append(row_values)
        else:
            failed_rows.append(row_values)
            failed_errors.append(error_msg)

    # Optional corrected timestamp CSV
    if args.drop_invalid_timestamps:
        valid_indices = [r[0] for r in results if r[2]]
        filtered_csv = csv.iloc[valid_indices]

        if args.csv_path.endswith("timestamp.csv"):
            corrected_path = args.csv_path.replace("timestamp.csv", "correct_timestamp.csv")
        else:
            base, ext = os.path.splitext(args.csv_path)
            corrected_path = f"{base}_corrected{ext}"

        filtered_csv.to_csv(corrected_path, index=False)
        print(f"Corrected timestamp file saved to '{corrected_path}'")

    columns = csv.columns

    # Save successful clips CSV
    if success_rows:
        success_df = pd.DataFrame(success_rows, columns=columns)

        # Keep output CSV cleaner, matching your earlier style
        for col in ["timestamp_start", "timestamp_end", "frame_start", "frame_end"]:
            if col in success_df.columns:
                success_df = success_df.drop(columns=[col])

        success_df.to_csv(args.csv_save_path, index=False)
        print(f"Saved {len(success_df)} successful clip(s) to {args.csv_save_path}.")
    else:
        print("No successful clips were generated.")

    # Save failed CSV
    if failed_rows:
        base, ext = os.path.splitext(args.csv_save_path)
        failed_csv_path = f"{base}_failed{ext}"

        failed_df = pd.DataFrame(failed_rows, columns=columns)
        failed_df["error"] = failed_errors
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"Saved {len(failed_df)} failed record(s) to {failed_csv_path}.")


if __name__ == "__main__":
    main()