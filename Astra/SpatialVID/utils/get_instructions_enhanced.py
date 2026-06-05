import argparse
from math import sqrt
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd
from multiprocessing import Manager
import concurrent.futures
import queue
from tqdm import tqdm
import json
from collections import defaultdict, Counter
import itertools


def filter_poses(poses_array, alpha):
    """
    Smooth pose sequence using Exponential Moving Average (EMA).
    - Position: Standard EMA
    - Quaternion: EMA with hemisphere check (shortest path interpolation)
    """
    positions = poses_array[:, :3]
    quaternions = poses_array[:, 3:]

    filtered_pos = np.zeros_like(positions)
    filtered_quat = np.zeros_like(quaternions)

    # Initialize with first frame
    filtered_pos[0], filtered_quat[0] = positions[0], quaternions[0]

    for i in range(1, len(poses_array)):
        # Position smoothing
        filtered_pos[i] = alpha * positions[i] + \
            (1 - alpha) * filtered_pos[i-1]

        # Quaternion smoothing with hemisphere correction
        last_q, curr_q = filtered_quat[i-1], quaternions[i]
        if np.dot(last_q, curr_q) < 0:  # Flip to shortest interpolation path
            curr_q = -curr_q

        interp_q = (1 - alpha) * last_q + alpha * curr_q
        filtered_quat[i] = interp_q / \
            np.linalg.norm(interp_q)  # Keep unit quaternion

    return np.hstack([filtered_pos, filtered_quat])


def poses_to_multi_instructions(poses_array, translation_thresh, rotation_thresh_deg, interval=1):
    """
    Convert pose sequence to motion instructions (e.g., Dolly, Pan).
    Calculates pose difference between frame i and i+interval (convolution-like).
    """
    # Convert to (position, Rotation object) pairs
    poses = [(row[:3], R.from_quat(row[3:])) for row in poses_array]
    command_seq = []

    # Adjust thresholds by interval (scaling for longer gaps)
    rotation_thresh_deg *= sqrt(interval) / 1.8
    rotation_thresh_rad = np.deg2rad(rotation_thresh_deg)
    translation_thresh *= sqrt(interval)
    stride = int(sqrt(interval) + 1)

    i = 0
    while True:
        if i + interval >= len(poses):  # Ensure valid frame pair
            break

        # Calculate relative motion (local coordinate system)
        pos_t_w2c, rot_t_w2c = poses[i]
        pos_t1_w2c, rot_t1_w2c = poses[i+interval]
        delta_rot = rot_t1_w2c * rot_t_w2c.inv()

        pos_t_c2w = rot_t_w2c.inv().apply(pos_t_w2c)
        pos_t1_c2w = rot_t1_w2c.inv().apply(pos_t1_w2c)
        local_delta_pos = rot_t_w2c.inv().apply(pos_t1_c2w - pos_t_c2w)

        dx, dy, dz = local_delta_pos
        yaw, pitch, roll = delta_rot.as_euler(
            "yxz")  # Yaw:Pan, Pitch:Tilt, Roll:Rotate

        instructions = []
        # Translation commands
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

        # Rotation commands
        if yaw > rotation_thresh_rad:
            instructions.append("Pan Left")
        elif yaw < -rotation_thresh_rad:
            instructions.append("Pan Right")
        if pitch > rotation_thresh_rad:
            instructions.append("Tilt Down")
        elif pitch < -rotation_thresh_rad:
            instructions.append("Tilt Up")
        if roll > rotation_thresh_rad:
            instructions.append("Roll CCW")
        elif roll < -rotation_thresh_rad:
            instructions.append("Roll CW")

        command_seq.append(instructions if instructions else ["Stay"])
        i += stride
    return command_seq


def calculate_relative_scale(total_distance, num_poses, f_translation, min_threshold=0.001):
    """
    Calculate relative translation threshold (dynamic scaling by total motion).
    """
    if num_poses <= 1:
        return min_threshold
    base_scale = total_distance / num_poses  # Base scale per frame
    return max(base_scale / f_translation, min_threshold)


def voter(args, row, interval, alpha):
    """
    Process single video with specific (interval, alpha) parameter pair.
    """
    # Locate pose file
    npy_path = os.path.join(
        args.dir_path, row["id"], "reconstructions", "poses.npy"
    )

    try:
        raw_poses = np.load(npy_path)
        filtered_poses = filter_poses(raw_poses, alpha)

        # Calculate dynamic thresholds
        translation_thresh = calculate_relative_scale(
            row["moveDist"], len(
                filtered_poses), args.f_translation, args.min_threshold_translation
        )
        rotation_thresh = args.rotation_threshold

        return poses_to_multi_instructions(
            filtered_poses, translation_thresh, rotation_thresh, interval
        )
    except Exception as e:
        print(f"Error processing {row['id']}: {e}")
        return None


def collect_all_results(args, row, param_combinations):
    """Collect instruction results for all (interval, alpha) pairs."""
    results = []
    for interval, alpha in param_combinations:
        res = voter(args, row, interval, alpha)
        if res is not None:
            results.append(res)
    return results


# ------------------------------ Voting Logic ------------------------------
def get_mutually_exclusive_groups():
    """Return groups of conflicting instructions (cannot coexist)."""
    return [
        ["Dolly In", "Dolly Out"], ["Truck Left", "Truck Right"],
        ["Pedestal Up", "Pedestal Down"], ["Pan Left", "Pan Right"],
        ["Tilt Up", "Tilt Down"], ["Roll CW", "Roll CCW"]
    ]


def remove_conflicting_instructions(instructions, conflict_groups):
    """Remove conflicting instructions (keep higher-voted ones)."""
    selected = []
    selected_set = set()
    for inst, count in instructions:
        conflict = False
        for group in conflict_groups:
            if inst in group and any(s in group for s in selected_set):
                conflict = True
                break
        if not conflict:
            selected.append((inst, count))
            selected_set.add(inst)
    return selected


def smart_instruction_selection(non_conflicting_inst):
    """
    Smart instruction selection based on vote distribution:
    - Keep leading votes (3x threshold for断层)
    - Max 4 instructions
    - Prioritize non-"Stay" commands
    """
    if not non_conflicting_inst:
        return ["Stay"]
    if len(non_conflicting_inst) == 1:
        return [non_conflicting_inst[0][0]]

    # Separate Stay and other instructions
    stay = [i for i in non_conflicting_inst if i[0] == "Stay"]
    others = [i for i in non_conflicting_inst if i[0] != "Stay"]
    if not others:
        return ["Stay"]

    votes = [c for _, c in others]
    max_vote = votes[0]
    selected = []

    # Check for vote gap (3x threshold)
    if len(others) >= 2 and max_vote >= votes[1] * 3:
        selected = [i[0] for i in others if i[1] == max_vote]
    else:
        # Select up to 4 leading instructions
        gap_thresh = max_vote * 0.5
        selected = [i[0] for i in others if i[1] >= gap_thresh][:4]

    # Ensure minimum 2 instructions if no large gap
    if len(selected) < 2 and len(others) >= 2 and max_vote < votes[1] * 3:
        selected = [i[0] for i in others[:2]]

    return selected if selected else ["Stay"]


def collect_interval_based_votes(all_results, param_combinations):
    """
    Vote by time interval: collect all instructions covering (start_frame->end_frame).
    Handles overlapping segments from different (interval, alpha) pairs.
    """
    if not all_results:
        return {}

    # Get max frame covered by any parameter pair
    max_frames = 0
    for index, res in enumerate(all_results):
        interval = param_combinations[index][0]
        stride = int(sqrt(interval) + 1)
        if res:
            last_start = (len(res)-1) * stride
            max_frames = max(max_frames, last_start + interval)

    interval_votes = {}
    for start in range(max_frames):
        end = start + 1
        vote_counter = Counter()
        # Check all parameter results for coverage of (start->end)
        for res_index, res in enumerate(all_results):
            interval, _ = param_combinations[res_index]
            stride = int(sqrt(interval) + 1)
            for seg_index, seg in enumerate(res):
                seg_start = seg_index * stride
                seg_end = seg_start + interval
                # Check if segment covers target interval
                if seg_start <= start < seg_end and seg_start < end <= seg_end:
                    for inst in seg:
                        vote_counter[inst] += 1
        interval_votes[f"{start}->{end}"] = vote_counter
    return interval_votes


def vote_for_final_instructions(all_results, param_combinations=None):
    """Generate final instructions via voting (interval-based if possible)."""
    if not all_results:
        return []

    conflict_groups = get_mutually_exclusive_groups()
    final_seq = []

    # Use interval-based voting if parameters are provided
    if param_combinations and len(param_combinations) == len(all_results):
        interval_votes = collect_interval_based_votes(
            all_results, param_combinations)
        for key in sorted(interval_votes.keys(), key=lambda x: int(x.split('->')[0])):
            votes = interval_votes[key]
            if votes:
                sorted_inst = votes.most_common()
                non_conflict = remove_conflicting_instructions(
                    sorted_inst, conflict_groups)
                selected = smart_instruction_selection(non_conflict)
            else:
                selected = ["Stay"]
            final_seq.append(selected)
    else:
        # Fallback: frame-wise voting
        max_len = max(len(res) for res in all_results)
        for frame_index in range(max_len):
            votes = Counter()
            for res in all_results:
                if frame_index < len(res):
                    for inst in res[frame_index]:
                        votes[inst] += 1
            if votes:
                sorted_inst = votes.most_common()
                non_conflict = remove_conflicting_instructions(
                    sorted_inst, conflict_groups)
                selected = smart_instruction_selection(non_conflict)
            else:
                selected = ["Stay"]
            final_seq.append(selected)
    return final_seq


# ------------------------------ Main Workflow ------------------------------
def merge_consecutive_instructions(instructions):
    """Merge consecutive identical instruction lists (e.g., [A,A,A] → "0->3":[A])."""
    if not instructions:
        return {}
    merged = {}
    start, prev = 0, instructions[0]
    for i in range(1, len(instructions)):
        if instructions[i] != prev:
            merged[f"{start}->{i}"] = prev
            start, prev = i, instructions[i]
    merged[f"{start}->{len(instructions)}"] = prev  # Add final segment
    return merged


def process_single_row(args, row, param_combinations):
    # Skip if output exists
    out_file = os.path.join(args.dir_path, row['id'], "instructions.json")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        return

    # Collect results & vote
    all_results = collect_all_results(args, row, param_combinations)
    if not all_results:
        print(f"No valid results for {row['id']}")
        return
    final_inst = vote_for_final_instructions(all_results, param_combinations)
    merged_inst = merge_consecutive_instructions(final_inst)

    # Save to JSON
    with open(out_file, "w") as f:
        json.dump(merged_inst, f, ensure_ascii=False, indent=2)


def generate_param_combinations(args):
    """Generate all (interval, alpha) parameter pairs for grid search."""
    intervals = getattr(args, "intervals", [1, 3, 5])
    alphas = getattr(args, "alphas", [0.03, 0.05, 0.1])
    return list(itertools.product(intervals, alphas))


def worker(task_queue, args, param_combinations, pbar):
    """Parallel worker: process tasks from queue."""
    while True:
        try:
            index, row = task_queue.get(timeout=1)
            process_single_row(args, row, param_combinations)
        except queue.Empty:
            break
        task_queue.task_done()
        pbar.update(1)


def args_parser():
    parser = argparse.ArgumentParser(
        description="Enhanced Camera Pose Instruction Generator")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Input CSV path (The final_results.csv generated by evaluation.py)")
    parser.add_argument("--dir_path", type=str, required=True,
                        help="Annotation directory path")
    parser.add_argument("--intervals", type=int, nargs="+",
                        default=[1, 3, 5], help="Frame intervals for grid search")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.03, 0.05, 0.1], help="Smoothing factors for grid search")
    parser.add_argument("--f_translation", type=float,
                        default=1.1, help="Translation scale factor (>1)")
    parser.add_argument("--min_threshold_translation", type=float,
                        default=0.01, help="Min translation threshold")
    parser.add_argument("--rotation_threshold", type=float,
                        default=1.5, help="Fixed rotation threshold (degrees)")
    parser.add_argument("--num_workers", type=int,
                        default=8, help="Parallel workers count")
    parser.add_argument("--disable_parallel", action="store_true",
                        help="Disable parallel processing")
    return parser.parse_args()


def main():
    args = args_parser()
    csv = pd.read_csv(args.csv_path)

    param_combinations = generate_param_combinations(args)

    if args.disable_parallel:
        # Serial processing
        for index, row in tqdm(csv.iterrows(), total=len(csv), desc="Processing"):
            process_single_row(index, args, row, param_combinations)
    else:
        # Parallel processing
        manager = Manager()
        task_queue = manager.Queue()
        for index, row in csv.iterrows():
            task_queue.put((index, row))

        with tqdm(total=len(csv), desc="Processing") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                for _ in range(args.num_workers):
                    executor.submit(worker, task_queue, args,
                                    param_combinations, pbar)
                task_queue.join()


if __name__ == "__main__":
    main()
