"""
Video cutting utility using FFmpeg with GPU acceleration support.
"""

import argparse
import os
import queue
import concurrent.futures
from functools import partial
import pandas as pd
import subprocess
from scenedetect import FrameTimecode
from tqdm import tqdm


# Default FFmpeg installation path
FFMPEG_PATH = "/usr/local/bin/ffmpeg"

def get_ffmpeg_acceleration():
    """Auto-detect FFmpeg acceleration: NVIDIA GPU > CPU"""
    try:
        output = subprocess.check_output(
            [FFMPEG_PATH, "-encoders"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
        if "hevc_nvenc" in output:
            return "nvidia"
        else:
            return "cpu"
    except Exception as e:
        print(f"FFmpeg acceleration detection failed: {e}")
        return "cpu"

# Detect and set the acceleration type
ACCELERATION_TYPE = get_ffmpeg_acceleration()
print(f"FFmpeg acceleration type: {ACCELERATION_TYPE}")


def process_single_row(row, args, process_id):
    """
    Process a single video row to extract clip using FFmpeg.

    Returns:
        (row_values_list, valid, error_message)
    """
    video_path = row["video_path"]
    save_dir = args.video_save_dir

    # Skip resizing if video is already smaller (no upscaling)
    shorter_size = args.shorter_size
    if (shorter_size is not None) and ("height" in row) and ("width" in row):
        min_size = min(row["height"], row["width"])
        if min_size <= shorter_size:
            shorter_size = None

    # Parse timestamps
    try:
        seg_start = FrameTimecode(timecode=row["timestamp_start"], fps=row["fps"])
        seg_end = FrameTimecode(timecode=row["timestamp_end"], fps=row["fps"])
    except Exception as e:
        error_msg = f"Invalid timestamp for id={row.get('id', '?')}: {e}"
        print(error_msg)
        return row.values.tolist(), False, error_msg

    id = row["id"]
    save_path = os.path.join(save_dir, f"{id}.mp4")

    if os.path.exists(save_path):
        row["video_path"] = save_path
        return row.values.tolist(), True, ""

    # Check source video exists
    if not os.path.exists(video_path):
        error_msg = f"Source video not found: {video_path} (id={id})"
        print(error_msg)
        return row.values.tolist(), False, error_msg

    try:
        # Build FFmpeg command
        cmd = [FFMPEG_PATH, "-nostdin", "-y"]
        if ACCELERATION_TYPE == "nvidia":
            cmd += [
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-hwaccel_device", str(process_id % args.gpu_num),
            ]

        cmd += [
            "-i", video_path,
            "-ss", str(seg_start.get_seconds()),
            "-to", str(seg_end.get_seconds()),
        ]

        if ACCELERATION_TYPE == "nvidia":
            cmd += ["-c:v", "hevc_nvenc", "-preset", "fast"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "fast"]

        if args.target_fps is not None:
            cmd += ["-r", str(args.target_fps)]

        if shorter_size is not None:
            if ACCELERATION_TYPE == "nvidia":
                cmd += [
                    "-vf",
                    f"scale_cuda='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'",
                ]
            else:
                cmd += [
                    "-vf",
                    f"scale='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'",
                ]

        cmd += ["-map", "0:v", save_path]

        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)

        # Verify output file
        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            if os.path.exists(save_path):
                os.remove(save_path)
            error_msg = f"FFmpeg produced empty/missing output for id={id}"
            print(error_msg)
            return row.values.tolist(), False, error_msg

        row["video_path"] = save_path
        return row.values.tolist(), True, ""

    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8") if e.stderr else str(e)
        error_msg = f"FFmpeg failed for id={id}: {stderr_text}"
        print(error_msg)
        if os.path.exists(save_path):
            os.remove(save_path)
        return row.values.tolist(), False, error_msg

    except Exception as e:
        error_msg = f"Unexpected error for id={id}: {e}"
        print(error_msg)
        if os.path.exists(save_path):
            os.remove(save_path)
        return row.values.tolist(), False, error_msg


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cut video clips from source videos")
    parser.add_argument("--csv_path", type=str, help="Path to input CSV file")
    parser.add_argument(
        "--csv_save_path", type=str, required=True, help="Path to save CSV file"
    )
    parser.add_argument(
        "--video_save_dir", type=str, required=True,
        help="Directory to save video clips",
    )
    parser.add_argument(
        "--target_fps", type=int, default=None, help="Target fps of output clips"
    )
    parser.add_argument(
        "--shorter_size", type=int, default=None,
        help="Resize shorter side keeping aspect ratio",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--disable_parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--drop_invalid_timestamps", action="store_true",
        help="Drop rows with invalid timestamps",
    )
    parser.add_argument(
        "--gpu_num", type=int, default=1, help="Number of GPUs available"
    )
    return parser.parse_args()


def worker(task_queue, results_queue, args, process_id):
    """Worker function for parallel video processing"""
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        row_values, valid, error_msg = process_single_row(row, args, process_id)
        results_queue.put((index, row_values, valid, error_msg))
        task_queue.task_done()


def main():
    """Main function to process video cutting"""
    args = parse_args()
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"csv file '{csv_path}' not found. Exit.")
        return

    os.makedirs(args.video_save_dir, exist_ok=True)

    csv = pd.read_csv(args.csv_path)

    # Setup multiprocessing
    from multiprocessing import Manager

    manager = Manager()
    task_queue = manager.Queue()
    results_queue = manager.Queue()

    for index, row in csv.iterrows():
        task_queue.put((index, row))

    if args.disable_parallel:
        for index, row in tqdm(
            csv.iterrows(), total=len(csv), desc="Processing rows"
        ):
            row_values, valid, error_msg = process_single_row(row, args, process_id=0)
            results_queue.put((index, row_values, valid, error_msg))
    else:
        num_workers = args.num_workers if args.num_workers else os.cpu_count() or 1
        results = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            futures = []
            for id in range(num_workers):
                futures.append(
                    executor.submit(worker, task_queue, results_queue, args, id)
                )

            processed = 0
            total_tasks = len(csv)
            with tqdm(total=total_tasks, desc="Processing rows") as pbar:
                while processed < total_tasks:
                    try:
                        results.append(results_queue.get(timeout=1))
                        processed += 1
                        pbar.update(1)
                    except queue.Empty:
                        if all(f.done() for f in futures) and results_queue.empty():
                            break

            for future in futures:
                future.result()

    # Collect results
    if args.disable_parallel:
        results = []
    while not results_queue.empty():
        results.append(results_queue.get())

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

    if args.drop_invalid_timestamps:
        filtered_csv = csv.iloc[[r[0] for r in results if r[2]]]
        assert args.csv_path.endswith("timestamp.csv"), "Only support *timestamp.csv"
        corrected_path = args.csv_path.replace("timestamp.csv", "correct_timestamp.csv")
        filtered_csv.to_csv(corrected_path, index=False)
        print(f"Corrected timestamp file saved to '{corrected_path}'")

    # Save successful results
    columns = csv.columns
    if success_rows:
        success_df = pd.DataFrame(success_rows, columns=columns)
        for col in ["timestamp_start", "timestamp_end", "frame_start", "frame_end"]:
            if col in success_df.columns:
                success_df = success_df.drop(columns=[col])
        success_df.to_csv(args.csv_save_path, index=False)
        print(f"Saved {len(success_df)} successful clip(s) to {args.csv_save_path}.")

    # Save failed results: derive path from csv_save_path
    if failed_rows:
        base, ext = os.path.splitext(args.csv_save_path)
        failed_csv_path = f"{base}_failed{ext}"

        failed_df = pd.DataFrame(failed_rows, columns=columns)
        failed_df["error"] = failed_errors
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"Saved {len(failed_df)} failed record(s) to {failed_csv_path}.")

if __name__ == "__main__":
    main()
