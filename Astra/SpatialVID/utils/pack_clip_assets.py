"""
pack_clip_assets.py
------------------
This script unifies depth, RGB frames, intrinsics, extrinsics, etc. of a specified video clip into a single npz file for downstream 3D reconstruction or analysis.

Usage example:
    python pack_clip_assets.py --base_dir /path/to/HQ --clip_id group_xxxx/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx --height 328 --width 584

"""

import argparse
import numpy as np
import torch
from lietorch import SE3
import cv2
from read_depth import read_depth

def load_video(clip_path, indexes_path, height=720, width=1280):
    """
    Read video frames at specified indexes and resize to (height, width).
    Args:
        clip_path (str): Path to video file
        indexes_path (str): Path to frame indexes txt
        height (int): Output frame height
        width (int): Output frame width
    Returns:
        np.ndarray: (N, height, width, 3) RGB frames
    """
    indexes = []
    with open(indexes_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                indexes.append(int(parts[1]))
    print(f"Frame indexes: {indexes}")
    cap = cv2.VideoCapture(clip_path)
    frames = []
    for idx in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Frame at index {idx} could not be read.")
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def load_intrinsics(intrinsics_path, tgt_width=1024, tgt_height=576):
    """
    Read normalized intrinsics (n,4), convert to 3x3 matrix and scale to target resolution.
    Args:
        intrinsics_path (str): Path to intrinsics npy
        tgt_width (int): Target width
        tgt_height (int): Target height
    Returns:
        np.ndarray: (N, 3, 3) intrinsics matrices
    """
    intrinsics = np.load(intrinsics_path)
    intrinsics_3x3 = []
    for intrin in intrinsics:
        fx, fy, cx, cy = intrin
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)
        intrinsics_3x3.append(K)
    intrinsics_3x3 = np.array(intrinsics_3x3)
    intrinsics_3x3[:, 0, 0] *= tgt_width
    intrinsics_3x3[:, 1, 1] *= tgt_height
    intrinsics_3x3[:, 0, 2] *= tgt_width
    intrinsics_3x3[:, 1, 2] *= tgt_height
    return intrinsics_3x3

def main():
    """
    Main pipeline: load depth, RGB frames, intrinsics, extrinsics, and save as npz.
    """
    parser = argparse.ArgumentParser(description="Pack clip assets into a single npz file.")
    parser.add_argument('--base_dir', type=str, required=True, help='Root directory of HQ data')
    parser.add_argument('--group_id', type=int, required=False, help='Group ID, e.g. group_xxxx')
    parser.add_argument('--clip_id', type=str, required=True, help='Clip ID, e.g. xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx')
    parser.add_argument('--height', type=int, default=328, help='Output image height')
    parser.add_argument('--width', type=int, default=584, help='Output image width')
    parser.add_argument('--output', type=str, default='sgd_cvd_hr.npz', help='Output npz filename')
    args = parser.parse_args()

    # Path construction
    annotation_dir = f'{args.base_dir}/annotations/group_{args.group_id:04d}/{args.clip_id}'
    depth_path = f'{args.base_dir}/depths/group_{args.group_id:04d}/{args.clip_id}.zip'
    clip_path = f'{args.base_dir}/videos/group_{args.group_id:04d}/{args.clip_id}.mp4'
    intrinsics_path = f'{annotation_dir}/intrinsics.npy'
    extrinsics_path = f'{annotation_dir}/poses.npy'
    indexes_path = f'{annotation_dir}/indexes.txt'

    # Load intrinsics and extrinsics
    intrinsics = load_intrinsics(intrinsics_path, tgt_width=args.width, tgt_height=args.height)
    extrinsics = np.load(extrinsics_path)

    # Load and resize depth
    depth = np.clip(read_depth(depth_path), 1e-3, 1e2)  # (N, H, W)
    resized_depth = np.zeros((depth.shape[0], args.height, args.width), dtype=depth.dtype)
    for i in range(depth.shape[0]):
        resized_depth[i] = cv2.resize(depth[i], (args.width, args.height), interpolation=cv2.INTER_LINEAR)

    # Load RGB frames
    frames = load_video(clip_path, indexes_path, args.height, args.width)

    # Compute camera poses
    poses_th = torch.as_tensor(extrinsics, device="cpu").float()
    cam_c2w = SE3(poses_th).inv().matrix()
    K = intrinsics[0]
    K_o = torch.from_numpy(K).float()

    # Save as npz
    np.savez(
        args.output,
        images=frames,
        depths=resized_depth,
        intrinsic=K_o.detach().cpu().numpy(),
        cam_c2w=cam_c2w.detach().cpu().numpy(),
    )
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
