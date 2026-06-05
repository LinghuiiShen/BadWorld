from __future__ import annotations

import dataclasses
import os
import json
from pathlib import Path
from typing import Tuple, cast

import imageio.v3 as iio
import liblzfse
import numpy as np
import numpy as onp
import numpy.typing as onpt
import skimage.transform
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

class Record3dLoader_Customized_Megasam:
    """Helper for loading frames for Record3D captures directly from a NPZ file."""

    def __init__(self, npz_data: dict, conf_threshold: float = 1.0, foreground_conf_threshold: float = 0.1, no_mask: bool = False, xyzw=True, init_conf=False):
        # Assuming npz_data is a dictionary containing all the necessary arrays from the NPZ file
        self.K = np.expand_dims(npz_data['intrinsic'], 0)               # (3,3) -> (1,3,3) Intrinsic matrix
        self.K = np.repeat(self.K, npz_data['images'].shape[0], axis=0) # (1,3,3) -> (N,3,3)
        self.T_world_cameras = npz_data['cam_c2w']                      # (N,4,4) Camera poses (extrinsics)
        self.fps = 30  # Assuming a frame rate of 30
        self.conf_threshold = conf_threshold
        self.foreground_conf_threshold = foreground_conf_threshold
        self.no_mask = no_mask

        # Initialize the other parameters
        self.init_conf = init_conf
        
        # Read frames from the NPZ file
        self.images = npz_data['images']                                # (N,H,W,3) RGB images
        self.depths = npz_data['depths']                                # (N,H,W) Depth maps
        self.confidences = npz_data.get('conf', [])
        self.init_conf_data = npz_data.get('init_conf', [])
        self.masks = npz_data.get('enlarged_dynamic_mask', [])

        # Align all camera poses by the first frame
        T0 = self.T_world_cameras[len(self.T_world_cameras) // 2]  # First camera pose (4x4 matrix)
        T0_inv = np.linalg.inv(T0)  # Inverse of the first camera pose

        # Apply T0_inv to all camera poses
        self.T_world_cameras = np.matmul(T0_inv[np.newaxis, :, :], self.T_world_cameras)

    def num_frames(self) -> int:
        return len(self.images)

    def get_frame(self, index: int) -> Record3dFrame:
        # Read the depth for the given frame
        depth = self.depths[index]
        depth = depth.astype(np.float32)

        # Check if conf file exists, otherwise initialize with ones
        if len(self.confidences) == 0:
            conf = np.ones_like(depth, dtype=np.float32)
        else:
            conf = self.confidences[index]
            conf = np.clip(conf, 0.0001, 99999)

        # Check if init conf file exists, otherwise initialize with ones
        if len(self.init_conf_data) == 0:
            init_conf = conf
        else:
            init_conf = self.init_conf_data[index]
            init_conf = np.clip(init_conf, 0.0001, 99999)
        
        # Check if mask exists, otherwise initialize with zeros
        if len(self.masks) == 0:
            mask = np.ones_like(depth, dtype=bool)
        else:
            mask = self.masks[index] > 0  # Assuming mask is a binary image

        if self.no_mask:
            mask = np.ones_like(mask).astype(np.bool_)

        # Read RGB image
        rgb = self.images[index]
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        return Record3dFrame(
            K=self.K[index],
            rgb=rgb,
            depth=depth,
            mask=mask,
            conf=conf,
            init_conf=init_conf,
            T_world_camera=self.T_world_cameras[index],
            conf_threshold=self.conf_threshold,
            foreground_conf_threshold=self.foreground_conf_threshold,
        )


@dataclasses.dataclass
class Record3dFrame:
    """A single frame from a Record3D capture."""

    K: onpt.NDArray[onp.float32]
    rgb: onpt.NDArray[onp.uint8]
    depth: onpt.NDArray[onp.float32]
    mask: onpt.NDArray[onp.bool_]
    conf: onpt.NDArray[onp.float32]
    init_conf: onpt.NDArray[onp.float32]
    T_world_camera: onpt.NDArray[onp.float32]
    conf_threshold: float = 1.0
    foreground_conf_threshold: float = 0.1

    def get_point_cloud(
        self, downsample_factor: int = 1, bg_downsample_factor: int = 1,
    ) -> Tuple[onpt.NDArray[onp.float32], onpt.NDArray[onp.uint8], onpt.NDArray[onp.float32], onpt.NDArray[onp.uint8]]:
        rgb = self.rgb[::downsample_factor, ::downsample_factor]
        depth = skimage.transform.resize(self.depth, rgb.shape[:2], order=0)
        mask = cast(
            onpt.NDArray[onp.bool_],
            skimage.transform.resize(self.mask, rgb.shape[:2], order=0),
        )
        assert depth.shape == rgb.shape[:2]

        K = self.K
        T_world_camera = self.T_world_camera

        img_wh = rgb.shape[:2][::-1]

        grid = (
            np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), 2) + 0.5
        )
        grid = grid * downsample_factor
        conf_mask = self.conf > self.conf_threshold
        if self.init_conf is not None:
            fg_conf_mask = self.init_conf > self.foreground_conf_threshold
        else:
            fg_conf_mask = self.conf > self.foreground_conf_threshold
        # reshape the conf mask to the shape of the depth
        conf_mask = skimage.transform.resize(conf_mask, depth.shape, order=0)
        fg_conf_mask = skimage.transform.resize(fg_conf_mask, depth.shape, order=0)

        # Foreground points
        homo_grid = np.pad(grid[fg_conf_mask & mask], ((0, 0), (0, 1)), constant_values=1)
        local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
        dirs = np.einsum("ij,bj->bi", T_world_camera[:3, :3], local_dirs)
        points = (T_world_camera[:3, 3] + dirs * depth[fg_conf_mask & mask, None]).astype(np.float32)
        point_colors = rgb[fg_conf_mask & mask]

        # Background points
        bg_homo_grid = np.pad(grid[conf_mask & ~mask], ((0, 0), (0, 1)), constant_values=1)
        bg_local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), bg_homo_grid)
        bg_dirs = np.einsum("ij,bj->bi", T_world_camera[:3, :3], bg_local_dirs)
        bg_points = (T_world_camera[:3, 3] + bg_dirs * depth[conf_mask & ~mask, None]).astype(np.float32)
        bg_point_colors = rgb[conf_mask & ~mask]

        if bg_downsample_factor > 1 and bg_points.shape[0] > 0:
            indices = np.random.choice(
                bg_points.shape[0],
                size=bg_points.shape[0] // bg_downsample_factor,
                replace=False
            )
            bg_points = bg_points[indices]
            bg_point_colors = bg_point_colors[indices]

        return points, point_colors, bg_points, bg_point_colors
