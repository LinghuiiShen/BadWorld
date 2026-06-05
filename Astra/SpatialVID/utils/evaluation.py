"""
Camera trajectory evaluation utility with anomaly detection and motion analysis.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import concurrent.futures
import multiprocessing as mp
from multiprocessing import Manager
import queue
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

# Import mask utility functions
from expand_npz import expand


def load_file(cam_pos_file, mask_file, device):
    """Load camera parameters and dynamic masks from files"""
    try:
        # Load camera parameters and split into position and rotation
        params = torch.from_numpy(np.load(cam_pos_file)).float().to(device)
        cam_pos = params[:, :3]  # Position coordinates
        cam_rotate = params[:, 3:]  # Rotation quaternions
        time_steps = params.shape[0]

        # Load and expand dynamic masks
        masks = torch.from_numpy(expand(np.load(mask_file))).to(device)
    except FileNotFoundError:
        print(f"Error: File not found - {cam_pos_file}")
        exit()
    except Exception as e:
        print(f"Error processing {cam_pos_file}: {e}")
        exit()

    return cam_pos, cam_rotate, time_steps, masks


def anomaly_detection(cam_pos, time_steps, threshold, device):
    """Detect trajectory anomalies using linear prediction with acceleration"""
    if time_steps < 4:
        return True  # Not enough data

    preds = torch.zeros((time_steps, 3), dtype=torch.float32, device=device)
    error_count = 0

    # Linear prediction with acceleration
    for t in range(0, time_steps - 3):
        # Calculate velocity and acceleration
        v1 = cam_pos[t + 2] - cam_pos[t + 1]
        v2 = cam_pos[t + 1] - cam_pos[t]
        acceleration = v1 - v2

        # Predict next position
        preds[t + 3] = cam_pos[t + 2] + v1 + 0.5 * acceleration

        # Check prediction error
        error = torch.sqrt(torch.sum((preds[t + 3] - cam_pos[t + 3]) ** 2))
        if error > 0.03:
            error_count += 1
            if error_count >= threshold:
                return True
        else:
            error_count = 0

    return False


def move_distance(cam_pos, time_steps, device):
    """Calculate total movement distance and classify into levels"""
    total_distance = torch.tensor(0., dtype=torch.float32, device=device)

    # Distance thresholds for classification
    thresholds = [0.08, 0.28, 0.92, 2.41]
    
    # Calculate cumulative distance
    for i in range(0, time_steps - 1):
        total_distance += torch.norm(cam_pos[i + 1] - cam_pos[i])

    # Determine movement level
    distance_val = total_distance.item()
    level = sum(1 for threshold in thresholds if distance_val >= threshold)

    return distance_val, level


def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    # Extract components (q in [x, y, z, w] format)
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]

    # Quaternion multiplication
    matrix = torch.tensor([
        [w1, -z1, y1, x1],
        [z1, w1, -x1, y1],
        [-y1, x1, w1, z1],
        [-x1, -y1, -z1, w1]
    ], dtype=q1.dtype, device=q1.device)

    vector = torch.tensor([x2, y2, z2, w2], dtype=q2.dtype, device=q2.device)
    result = torch.matmul(matrix, vector)

    return result


def rotation_angle(cam_rotate, time_steps, device):
    """Calculate total rotation angle between consecutive frames"""
    total_radians = torch.tensor(0.0, device=device)

    for i in range(0, time_steps - 1):
        q1 = cam_rotate[i]
        q2 = cam_rotate[i + 1]

        # Calculate relative rotation
        q1_inverse = torch.stack([-q1[0], -q1[1], -q1[2], q1[3]], dim=0)
        q_relative = quaternion_multiply(q2, q1_inverse)
        w = torch.clamp(q_relative[3], -1.0, 1.0)

        # Convert to angle
        rotation_angle_rad = 2 * torch.arccos(w)
        total_radians += rotation_angle_rad

    return total_radians.item()


def trajectory_turns(cam_pos, time_steps, device, threshold=0.45):
    """Detect significant turns in camera trajectory"""
    if time_steps < 3:
        return [], 0

    angles = []
    # Calculate angles between trajectory segments
    for t in range(1, time_steps - 1):
        v1 = cam_pos[t] - cam_pos[0]
        v2 = cam_pos[time_steps - 1] - cam_pos[t]

        # Avoid division by zero
        v1_norm = torch.norm(v1)
        v2_norm = torch.norm(v2)
        if v1_norm < 1e-8 or v2_norm < 1e-8:
            continue

        # Calculate angle between vectors
        cos_theta = torch.dot(v1, v2) / (v1_norm * v2_norm)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        angle = torch.arccos(cos_theta)
        angles.append(angle.item())

    # Smooth and find peaks
    angles = gaussian_filter(angles, sigma=5)
    peaks, _ = find_peaks(angles, height=threshold, distance=5)
    peaks_values = [angles[i] for i in peaks]

    # Include maximum angle if significant
    max_angle = max(angles)
    if max_angle > threshold and max_angle not in peaks_values:
        peaks_values.append(max_angle)

    return len(peaks_values)


def dynamic_ratio(masks):
    """Calculate ratio of dynamic pixels in video frames"""
    # Downsample for efficiency
    masks = masks[::5, :, :]
    
    dynamic_pixels = torch.sum(masks)
    total_pixels = masks.shape[1] * masks.shape[2] * masks.shape[0]
    
    return (dynamic_pixels / total_pixels).item()


def process_single_row(row, index, args, device):
    """Process a single video row to extract trajectory metrics"""
    video_id = row['id']
    rec_path = os.path.join(args.dir_path, video_id, "reconstructions")
    cam_pos_file = os.path.join(rec_path, "poses.npy")
    mask_file = os.path.join(rec_path, "dyn_masks.npz")

    # Check file existence
    if not os.path.exists(cam_pos_file) or not os.path.exists(mask_file):
        print(f"File not found: {cam_pos_file} or {mask_file}")
        return False, False, -1, -1, -1, -1, -1

    # Load and process data
    cam_pos, cam_rotate, time_steps, masks = load_file(cam_pos_file, mask_file, device)

    # Calculate metrics
    anomaly = anomaly_detection(cam_pos, time_steps, args.anomaly_threshold, device)
    move_dist, dist_level = move_distance(cam_pos, time_steps, device)
    rot_angle = rotation_angle(cam_rotate, time_steps, device)
    traj_turns = trajectory_turns(cam_pos, time_steps, device)
    dyn_ratio = dynamic_ratio(masks)

    return True, anomaly, move_dist, dist_level, rot_angle, traj_turns, dyn_ratio


def worker(task_queue, result_queue, args, worker_id):
    """Worker function for parallel processing"""
    # Assign GPU based on worker ID
    device = torch.device(
        f"cuda:{worker_id % args.gpu_num}"
        if torch.cuda.is_available() else "cpu"
    )

    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break

        result = process_single_row(row, index, args, device)
        result_queue.put((index, result))
        task_queue.task_done()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Camera Trajectory Evaluation")
    parser.add_argument("--csv_path", type=str, help="Path to input CSV file")
    parser.add_argument("--dir_path", type=str, default="./outputs", help="Base directory with reconstruction data")
    parser.add_argument("--output_path", type=str, default="./outputs/evaluation_results.csv", help="Output CSV path")
    parser.add_argument("--anomaly_threshold", type=int, default=2, help="Anomaly detection threshold")
    parser.add_argument('--gpu_num', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--disable_parallel", action="store_true", help="Disable parallel processing")
    return parser.parse_args()


if __name__ == "__main__":
    # Setup multiprocessing
    mp.set_start_method('spawn')
    args = parse_args()

    # Load input data
    df = pd.read_csv(args.csv_path)

    results = []
    if args.disable_parallel:
        # Sequential processing
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
            device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
            result = process_single_row(row, index, args, device)
            results.append((index, result))
    else:
        # Parallel processing
        manager = Manager()
        task_queue = manager.Queue()

        # Add tasks to queue
        for index, row in df.iterrows():
            task_queue.put((index, row))

        result_queue = manager.Queue()

        # Run workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for worker_id in range(args.num_workers):
                futures.append(executor.submit(worker, task_queue, result_queue, args, worker_id))

            processed = 0
            total_tasks = len(df)
            with tqdm(total=total_tasks, desc="Processing videos") as pbar:
                while processed < total_tasks:
                    try:
                        index, result = result_queue.get(timeout=1)
                        results.append((index, result))
                        processed += 1
                        pbar.update(1)
                    except queue.Empty:
                        if all(f.done() for f in futures) and result_queue.empty():
                            break

            for future in futures:
                future.result()

        # Collect results
        while not result_queue.empty():
            index, result = result_queue.get()
            results.append((index, result))

    # Sort and save results
    results.sort(key=lambda x: x[0])

    df['success'] = [result[1][0] for result in results]
    df['anomaly'] = [result[1][1] for result in results]
    df['moveDist'] = [result[1][2] for result in results]
    df['distLevel'] = [result[1][3] for result in results]
    df['rotAngle'] = [result[1][4] for result in results]
    df['trajTurns'] = [result[1][5] for result in results]
    df['dynamicRatio'] = [result[1][6] for result in results]

    df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")
