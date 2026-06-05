"""
This module processes camera pose sequences and generates movement instructions.
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd
from multiprocessing import Manager
import concurrent.futures
import queue
from tqdm import tqdm
import json


def filter_poses(poses_array, alpha):
    """
    Filter pose sequences using exponential moving average.
    - Position: Exponential moving average (EMA)
    - Orientation (quaternion): NLERP-based EMA with hemisphere flip handling

    Args:
        poses_array: Array of poses [position(3) + quaternion(4)]
        alpha: Smoothing factor (0 < alpha < 1)

    Returns:
        Filtered pose array with same shape as input
    """

    positions = poses_array[:, :3]
    quaternions = poses_array[:, 3:]

    filtered_positions = np.zeros_like(positions)
    filtered_quaternions = np.zeros_like(quaternions)

    # Initialize with first frame
    filtered_positions[0] = positions[0]
    filtered_quaternions[0] = quaternions[0]

    for i in range(1, len(poses_array)):
        filtered_positions[i] = (
            alpha * positions[i] + (1 - alpha) * filtered_positions[i - 1]
        )

        # quaternion filtering with hemisphere check
        last_q = filtered_quaternions[i - 1]
        current_q = quaternions[i]

        # 1. Check hemisphere to ensure interpolation takes the "shortest path"
        if np.dot(last_q, current_q) < 0:
            current_q = -current_q

        # 2. Linear interpolation
        interp_q = (1 - alpha) * last_q + alpha * current_q

        # 3. Re-normalize to ensure unit quaternion
        filtered_quaternions[i] = interp_q / np.linalg.norm(interp_q)

    return np.hstack([filtered_positions, filtered_quaternions])


def poses_to_multi_instructions(poses_array, translation_thresh, rotation_thresh_deg):
    """
    Convert camera pose sequence to concurrent movement instruction sequence.
    """

    # Convert NumPy array to Scipy Rotation objects for easier computation
    poses = []
    for row in poses_array:
        pos = row[:3]
        rot = R.from_quat(row[3:])
        poses.append((pos, rot))

    command_sequence = []
    rotation_thresh_rad = np.deg2rad(rotation_thresh_deg)

    for i in range(len(poses) - 1):
        # Calculate local relative movement
        pos_t_w2c, rot_t_w2c = poses[i]
        pos_t1_w2c, rot_t1_w2c = poses[i+1]
        delta_rot = rot_t1_w2c * rot_t_w2c.inv()

        pos_t_c2w = rot_t_w2c.inv().apply(pos_t_w2c)
        pos_t1_c2w = rot_t1_w2c.inv().apply(pos_t1_w2c)
        local_delta_pos = rot_t_w2c.inv().apply(pos_t1_c2w - pos_t_c2w)

        dx, dy, dz = local_delta_pos
        euler_angles_rad = delta_rot.as_euler(
            "yxz"
        )  # 'y' for yaw, 'x' for pitch, 'z' for roll
        yaw_change, pitch_change, roll_change = euler_angles_rad

        instructions = []

        # Translation movements
        if dz < -translation_thresh:
            instructions.append("Dolly Out")
        elif dz > translation_thresh:
            instructions.append("Dolly In")
        if dx > translation_thresh:
            instructions.append("Truck Right")
        elif dx < -translation_thresh:
            instructions.append("Truck Left")
        if dy > translation_thresh:
            instructions.append("Pedestal Down")
        elif dy < -translation_thresh:
            instructions.append("Pedestal Up")

        # Rotation movements
        if yaw_change > rotation_thresh_rad:
            instructions.append("Pan Left")
        elif yaw_change < -rotation_thresh_rad:
            instructions.append("Pan Right")

        if pitch_change > rotation_thresh_rad:
            instructions.append("Tilt Down")
        elif pitch_change < -rotation_thresh_rad:
            instructions.append("Tilt Up")

        if roll_change > rotation_thresh_rad:
            instructions.append("Roll CCW")
        elif roll_change < -rotation_thresh_rad:
            instructions.append("Roll CW")

        if not instructions:
            instructions.append("Stay")

        command_sequence.append(instructions)

    return command_sequence


def process_single_row(args, row):
    """Process a single video row to generate camera movement instructions."""
    npy_path = os.path.join(args.dir_path, row["id"], "reconstructions", "poses.npy")

    # Load and subsample poses, then apply filtering
    raw_poses = np.load(npy_path)[:: args.interval]
    filtered_poses = filter_poses(raw_poses, alpha=args.alpha)

    # Generate movement instructions
    instructions = poses_to_multi_instructions(
        filtered_poses, args.translation_threshold, args.rotation_threshold
    )

    json_file = os.path.join(args.dir_path, row["id"], "instructions.json")
    if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
        return

    # Merge consecutive identical instructions
    merged_instructions = {}
    start = 0
    prev_cmd = instructions[0]
    for i in range(1, len(instructions)):
        if instructions[i] == prev_cmd:
            continue
        else:
            key = f"{start}->{i}"
            merged_instructions[key] = prev_cmd
            start = i
            prev_cmd = instructions[i]
    # Add final segment
    key = f"{start}->{len(instructions)}"
    merged_instructions[key] = prev_cmd

    # Save instructions to JSON file
    with open(json_file, "w") as f:
        json.dump(merged_instructions, f, ensure_ascii=False, indent=2)


def worker(task_queue, args, pbar):
    """Worker function for parallel processing of video rows."""
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break

        process_single_row(args, row)

        task_queue.task_done()
        pbar.update(1)


def args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, default="outputs.csv", help="Path to the input CSV file"
    )
    parser.add_argument("--dir_path", type=str, default="./outputs")
    parser.add_argument(
        "--interval", type=int, default=2, help="Interval for computing instructions"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Smoothing factor for filtering (0 < alpha < 1)",
    )
    parser.add_argument(
        "--translation_threshold",
        type=float,
        default=0.02,
        help="Translation threshold for command generation",
    )
    parser.add_argument(
        "--rotation_threshold",
        type=float,
        default=0.5,
        help="Rotation threshold for command generation",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of parallel workers"
    )
    parser.add_argument(
        "--disable_parallel", action="store_true", help="Disable parallel processing"
    )
    return parser.parse_args()


def main():
    args = args_parser()
    csv = pd.read_csv(args.csv_path)

    if args.disable_parallel:
        # Sequential processing
        for index, row in tqdm(csv.iterrows(), total=len(csv)):
            process_single_row(args, row)
    else:
        # Parallel processing using ThreadPoolExecutor
        manager = Manager()
        task_queue = manager.Queue()
        for index, row in csv.iterrows():
            task_queue.put((index, row))

        with tqdm(total=len(csv), desc="Finished tasks") as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.num_workers
            ) as executor:
                futures = []
                for _ in range(args.num_workers):
                    futures.append(executor.submit(worker, task_queue, args, pbar))
                for future in concurrent.futures.as_completed(futures):
                    future.result()


if __name__ == "__main__":
    main()
