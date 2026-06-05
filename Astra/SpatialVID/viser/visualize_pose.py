#!/usr/bin/env python3
"""
Camera Pose Visualization Module

This module provides comprehensive tools for visualizing camera poses and trajectories
in 3D space using Plotly. It supports both static and animated visualizations with
automatic camera view optimization.

Adapted from: https://huggingface.co/datasets/nvidia/dynpose-100k/blob/main/scripts/visualize_pose.py
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objs as go
import plotly.io as pio
from tqdm import tqdm
import einops
import torch

# Use non-interactive backend for matplotlib to avoid display issues
matplotlib.use("agg")


class Pose:
    """
    A class of operations on camera poses (numpy arrays with shape [...,3,4]).
    Each [3,4] camera pose takes the form of [R|t].
    """

    def __call__(self, R=None, t=None):
        """
        Construct a camera pose from the given rotation matrix R and/or translation vector t.
        """
        assert R is not None or t is not None
        if R is None:
            if not isinstance(t, np.ndarray):
                t = np.array(t)
            R = np.eye(3).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, np.ndarray):
                R = np.array(R)
            t = np.zeros(R.shape[:-1])
        else:
            if not isinstance(R, np.ndarray):
                R = np.array(R)
            if not isinstance(t, np.ndarray):
                t = np.array(t)
        assert R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3)
        R = R.astype(np.float32)
        t = t.astype(np.float32)
        pose = np.concatenate([R, t[..., None]], axis=-1)  # [...,3,4]
        assert pose.shape[-2:] == (3, 4)
        return pose

    def invert(self, pose, use_inverse=False):
        """
        Invert a camera pose.
        """
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = np.linalg.inv(R) if use_inverse else R.transpose(0, 2, 1)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        """
        Compose a sequence of poses together.
        pose_new(x) = poseN o ... o pose2 o pose1(x)
        """
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        """
        Compose two poses together.
        """
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new

    def scale_center(self, pose, scale):
        """
        Scale the camera center from the origin.
        0 = R@c+t --> c = -R^T@t (camera center in world coordinates)
        0 = R@(sc)+t' --> t' = -R@(sc) = -R@(-R^T@st) = st
        """
        R, t = pose[..., :3], pose[..., 3:]
        pose_new = np.concatenate([R, t * scale], axis=-1)
        return pose_new


def to_hom(X):
    """
    Convert points to homogeneous coordinates by appending ones.
    """
    X_hom = np.concatenate([X, np.ones_like(X[..., :1])], axis=-1)
    return X_hom


def cam2world(X, pose):
    """
    Transform points from camera coordinates to world coordinates.
    """
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom @ pose_inv.transpose(0, 2, 1)


def get_camera_mesh(pose, depth=1):
    """
    Create a 3D mesh representation of camera frustums for visualization.
    """
    # Define camera frustum geometry: 4 corners of image plane + camera center
    vertices = (
        np.array(
            [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1], [0, 0, 0]]
        )
        * depth
    )  # Shape: [5, 3] - 4 image plane corners + camera center

    # Define triangular faces for the camera frustum mesh
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    )  # Shape: [6, 3] - 6 triangular faces forming the pyramid

    # Transform vertices from camera space to world space
    vertices = cam2world(vertices[None], pose)  # Shape: [N, 5, 3]

    # Create wireframe lines connecting: corners -> center -> next corner
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]  # Shape: [N, 10, 3]

    return vertices, faces, wireframe


# def merge_xyz_indicators_plotly(xyz):
#     """Merge xyz coordinate indicators for plotly visualization."""
#     xyz = xyz[:, [[-1, 0], [-1, 1], [-1, 2]]]  # [N,3,2,3]
#     xyz_0, xyz_1 = unbind_np(xyz, axis=2)  # [N,3,3]
#     xyz_dummy = xyz_0 * np.nan
#     xyz_merged = np.stack([xyz_0, xyz_1, xyz_dummy], axis=2)  # [N,3,3,3]
#     xyz_merged = xyz_merged.reshape(-1, 3)
#     return xyz_merged


# def get_xyz_indicators(pose, length=0.1):
#     """Get xyz coordinate axis indicators for a camera pose."""
#     xyz = np.eye(4, 3)[None] * length
#     xyz = cam2world(xyz, pose)
#     return xyz


def merge_wireframes_plotly(wireframe):
    """
    Merge camera wireframes for efficient Plotly visualization.
    """
    wf_dummy = wireframe[:, :1] * np.nan  # Create NaN separators
    wireframe_merged = np.concatenate([wireframe, wf_dummy], axis=1).reshape(-1, 3)
    return wireframe_merged


def merge_meshes(vertices, faces):
    """
    Merge multiple camera meshes into a single mesh for efficient rendering.
    """
    mesh_N, vertex_N = vertices.shape[:2]
    # Adjust face indices for each mesh by adding vertex offset
    faces_merged = np.concatenate([faces + i * vertex_N for i in range(mesh_N)], axis=0)
    # Flatten all vertices into single array
    vertices_merged = vertices.reshape(-1, vertices.shape[-1])
    return vertices_merged, faces_merged


def unbind_np(array, axis=0):
    """
    Split numpy array along specified axis into a list of arrays.
    """
    if axis == 0:
        return [array[i, :] for i in range(array.shape[0])]
    elif axis == 1 or (len(array.shape) == 2 and axis == -1):
        return [array[:, j] for j in range(array.shape[1])]
    elif axis == 2 or (len(array.shape) == 3 and axis == -1):
        return [array[:, :, j] for j in range(array.shape[2])]
    else:
        raise ValueError("Invalid axis. Use 0 for rows, 1 for columns, or 2 for depth.")


def plotly_visualize_pose(
    poses, vis_depth=0.5, xyz_length=0.5, center_size=2, xyz_width=5, mesh_opacity=0.05
):
    """
    Create comprehensive Plotly visualization traces for camera poses.
    """
    N = len(poses)

    # Calculate camera centers in world coordinates
    centers_cam = np.zeros([N, 1, 3])  # Camera centers in camera space (origin)
    centers_world = cam2world(centers_cam, poses)  # Transform to world space
    centers_world = centers_world[:, 0]  # Remove extra dimension [N, 3]

    # Generate camera frustum geometry
    vertices, faces, wireframe = get_camera_mesh(poses, depth=vis_depth)

    # Merge all camera meshes into single arrays for efficient rendering
    vertices_merged, faces_merged = merge_meshes(vertices, faces)
    wireframe_merged = merge_wireframes_plotly(wireframe)

    # Extract x, y, z coordinates for Plotly
    wireframe_x, wireframe_y, wireframe_z = unbind_np(wireframe_merged, axis=-1)
    centers_x, centers_y, centers_z = unbind_np(centers_world, axis=-1)
    vertices_x, vertices_y, vertices_z = unbind_np(vertices_merged, axis=-1)

    # Set up rainbow color mapping for trajectory progression
    color_map = plt.get_cmap("gist_rainbow")  # red -> yellow -> green -> blue -> purple
    center_color = []
    faces_merged_color = []
    wireframe_color = []

    # Determine quarter positions for emphasis (start, 1/3, 2/3, end)
    quarter_indices = set([0])  # Always include start
    if N >= 3:
        quarter_indices.add(N // 3)
        quarter_indices.add(2 * N // 3)
    quarter_indices.add(N - 1)  # Always include end

    # Apply colors with emphasis on key trajectory points
    for i in range(N):
        # Emphasize quarter positions with higher opacity and brightness
        is_quarter = i in quarter_indices
        alpha = 6.0 if is_quarter else 0.4  # Higher opacity for key points

        # Generate color from rainbow colormap
        r, g, b, _ = color_map(i / (N - 1))
        rgb = np.array([r, g, b]) * (1.2 if is_quarter else 0.8)  # Brighten key points
        rgba = np.concatenate([rgb, [alpha]])

        # Apply colors to all visualization elements
        wireframe_color += [rgba] * 11  # 11 line segments per camera wireframe
        center_color += [rgba]
        faces_merged_color += [rgba] * 6  # 6 triangular faces per camera frustum

    # Create Plotly trace objects
    plotly_traces = [
        # Camera wireframe outlines
        go.Scatter3d(
            x=wireframe_x,
            y=wireframe_y,
            z=wireframe_z,
            mode="lines",
            line=dict(color=wireframe_color, width=1),
            name="Camera Wireframes",
        ),
        # Camera center points
        go.Scatter3d(
            x=centers_x,
            y=centers_y,
            z=centers_z,
            mode="markers",
            marker=dict(color=center_color, size=center_size, opacity=1),
            name="Camera Centers",
        ),
        # Camera frustum mesh faces
        go.Mesh3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            i=[f[0] for f in faces_merged],
            j=[f[1] for f in faces_merged],
            k=[f[2] for f in faces_merged],
            facecolor=faces_merged_color,
            opacity=mesh_opacity,
            name="Camera Frustums",
        ),
    ]
    return plotly_traces


def compute_optimal_camera_view(poses):
    """
    Compute optimal camera view parameters to ensure the entire trajectory is visible
    and aesthetically pleasing.
    """
    # Calculate all camera positions in world coordinates
    centers_cam = np.zeros([len(poses), 1, 3])
    centers_world = cam2world(centers_cam, poses)[:, 0]

    # Compute bounding box of the trajectory
    min_coords = np.min(centers_world, axis=0)
    max_coords = np.max(centers_world, axis=0)
    ranges = max_coords - min_coords

    # Calculate trajectory center point
    trajectory_center = (min_coords + max_coords) / 2

    # Calculate maximum range for adaptive scaling
    max_range = np.max(ranges)

    # Set minimum range to avoid division by zero for very small trajectories
    if max_range < 1e-6:
        max_range = 1.0
        ranges = np.ones(3)

    # Calculate principal direction of trajectory using PCA (Principal Component Analysis)
    if len(centers_world) > 1:
        # Center the points by subtracting the mean
        centered_points = centers_world - trajectory_center

        # Compute covariance matrix for PCA
        cov_matrix = np.cov(centered_points.T)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Main direction is the first eigenvector (highest variance)
        main_direction = eigenvectors[:, 0]

        # Ensure main direction points towards trajectory's positive direction
        start_to_end = centers_world[-1] - centers_world[0]
        if np.dot(main_direction, start_to_end) < 0:
            main_direction = -main_direction

    else:
        # Default direction for single pose or insufficient data
        main_direction = np.array([1, 0, 0])

    # Calculate optimal camera distance
    # Based on trajectory range and field of view, using smaller factor for better screen filling
    fov_factor = (
        0.8  # Reduced field of view factor to make trajectory occupy more screen space
    )
    base_distance = max_range * fov_factor

    # Consider trajectory aspect ratio and adjust distance accordingly
    aspect_ratios = ranges / max_range
    distance_scale = 1.0 + 0.1 * np.std(
        aspect_ratios
    )  # Reduced distance adjustment magnitude
    camera_distance = base_distance * distance_scale

    # Calculate optimal camera position
    # Method 1: Diagonal viewing angle based on main direction
    up_vector = np.array([0, 0, 1])  # World up direction (Z-axis)

    # Adjust strategy if main direction is nearly vertical
    if abs(np.dot(main_direction, up_vector)) > 0.9:
        # Main direction is nearly vertical, use side view
        view_direction = np.cross(main_direction, np.array([1, 0, 0]))
        if np.linalg.norm(view_direction) < 0.1:
            view_direction = np.cross(main_direction, np.array([0, 1, 0]))
        view_direction = view_direction / np.linalg.norm(view_direction)
    else:
        # Calculate diagonal view direction perpendicular to main direction
        # Combine horizontal component of main direction with tilt angle
        horizontal_component = (
            main_direction - np.dot(main_direction, up_vector) * up_vector
        )
        horizontal_component = horizontal_component / (
            np.linalg.norm(horizontal_component) + 1e-8
        )

        # Add some tilt angles for better 3D perspective
        elevation_angle = np.pi / 6  # 30 degrees elevation angle
        azimuth_offset = np.pi / 4  # 45 degrees azimuth offset

        # Create tilted view direction for optimal 3D perspective
        view_direction = (
            horizontal_component * np.cos(azimuth_offset) * np.cos(elevation_angle)
            + np.cross(horizontal_component, up_vector)
            * np.sin(azimuth_offset)
            * np.cos(elevation_angle)
            + up_vector * np.sin(elevation_angle)
        )

    # Calculate camera eye position
    camera_eye = trajectory_center + view_direction * camera_distance

    # Fine-tune camera position to ensure entire trajectory is within view
    # Calculate vectors from camera position to all trajectory points
    view_vectors = centers_world - camera_eye
    view_distances = np.linalg.norm(view_vectors, axis=1)

    # Adjust camera distance moderately if some points are too close
    min_distance = camera_distance * 0.3  # Reduced minimum distance ratio
    if np.min(view_distances) < min_distance:
        distance_adjustment = min_distance / np.min(view_distances)
        # Limit adjustment magnitude to avoid excessive scaling
        distance_adjustment = min(
            distance_adjustment, 1.2
        )  # Further limit adjustment range
        camera_eye = (
            trajectory_center + view_direction * camera_distance * distance_adjustment
        )

    # Calculate adaptive parameters with appropriate proportions
    auto_vis_depth = max_range * 0.08  # Moderately reduced camera frustum size
    auto_center_size = max_range * 1.5  # Moderately reduced center point size

    # Ensure parameters are within reasonable bounds
    auto_vis_depth = max(0.01, min(auto_vis_depth, max_range * 0.2))
    auto_center_size = max(0.1, min(auto_center_size, max_range * 2.0))

    return {
        "camera_eye": camera_eye,
        "trajectory_center": trajectory_center,
        "auto_vis_depth": auto_vis_depth,
        "auto_center_size": auto_center_size,
        "max_range": max_range,
        "ranges": ranges,
        "main_direction": main_direction,
    }


def compute_multiple_camera_views(poses):
    """
    Compute multiple optimized camera view angles, providing different viewing options.
    """
    base_params = compute_optimal_camera_view(poses)

    trajectory_center = base_params["trajectory_center"]
    max_range = base_params["max_range"]
    main_direction = base_params["main_direction"]

    # Calculate multiple view options
    views = {}

    # 1. Best automatic view (original optimal view)
    views["optimal"] = base_params

    # 2. Top-down bird's eye view
    top_distance = max_range * 1.5  # Further reduced top-down view distance
    views["top"] = {
        **base_params,
        "camera_eye": trajectory_center + np.array([0, 0, top_distance]),
        "description": "Top-down view",
    }

    # 3. Side view perspective
    side_distance = max_range * 1.3  # Further reduced side view distance
    side_direction = np.cross(main_direction, np.array([0, 0, 1]))
    if np.linalg.norm(side_direction) < 0.1:
        side_direction = np.array([1, 0, 0])
    else:
        side_direction = side_direction / np.linalg.norm(side_direction)

    views["side"] = {
        **base_params,
        "camera_eye": trajectory_center + side_direction * side_distance,
        "description": "Side view",
    }

    # 4. Diagonal view (45-degree elevation)
    diagonal_distance = max_range * 1.4  # Further reduced diagonal view distance
    elevation = np.pi / 4  # 45 degrees elevation
    azimuth = np.pi / 4  # 45 degrees azimuth angle

    diagonal_direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )

    views["diagonal"] = {
        **base_params,
        "camera_eye": trajectory_center + diagonal_direction * diagonal_distance,
        "description": "Diagonal view (45° elevation)",
    }

    # 5. Trajectory start-oriented view
    if len(poses) > 1:
        start_to_center = trajectory_center - base_params["camera_eye"]
        start_distance = max_range * 1.2  # Further reduced start view distance
        start_direction = start_to_center / (np.linalg.norm(start_to_center) + 1e-8)

        views["trajectory_start"] = {
            **base_params,
            "camera_eye": trajectory_center + start_direction * start_distance,
            "description": "View from trajectory start direction",
        }

    # 6. Compact view - ensure entire trajectory is fully visible
    fit_distance = max_range * 0.6  # Very compact distance for close-up view
    fit_direction = np.array([0.7, 0.7, 0.5])  # Stable viewing direction
    fit_direction = fit_direction / np.linalg.norm(fit_direction)

    views["fit_all"] = {
        **base_params,
        "camera_eye": trajectory_center + fit_direction * fit_distance,
        "description": "Fit all trajectory in view",
    }

    return views


def add_view_selector_to_html(html_str, views):
    """
    Add interactive view selector to HTML visualization.

    This function injects JavaScript code into the HTML to provide an interactive
    interface for switching between different camera views and enabling auto-rotation.

    Args:
        html_str: Original HTML string containing the Plotly visualization
        views: Dictionary of view configurations

    Returns:
        str: Enhanced HTML string with view selector and controls
    """

    # Generate JavaScript code for view selector
    view_selector_js = """
    <div id="view-selector" style="position: fixed; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); font-family: Arial, sans-serif; font-size: 12px; z-index: 1000; min-width: 120px;">
        <button onclick="autoRotate()" style="background: #ffc107; color: black; border: none; padding: 8px 12px; border-radius: 4px; cursor: pointer; width: 100%;">Auto Rotate</button>
    </div>
    
    <script>
    // Pre-defined view configurations
    const views = {"""

    # Add view data to JavaScript
    for view_name, view_data in views.items():
        eye = view_data["camera_eye"]
        center = view_data["trajectory_center"]
        view_selector_js += f"""
        {view_name}: {{
            eye: {{x: {eye[0]:.6f}, y: {eye[1]:.6f}, z: {eye[2]:.6f}}},
            center: {{x: {center[0]:.6f}, y: {center[1]:.6f}, z: {center[2]:.6f}}},
            up: {{x: 0, y: 0, z: 1}}
        }},"""

    view_selector_js += """
    };
    
    let rotationInterval = null;
    
    function autoRotate() {
        if (rotationInterval) {
            clearInterval(rotationInterval);
            rotationInterval = null;
            return;
        }
        
        var plotlyDiv = document.querySelector('.plotly-graph-div');
        if (!plotlyDiv) return;
        
        var currentView = views.fit_all;
        var center = currentView.center;
        var radius = Math.sqrt(
            Math.pow(currentView.eye.x - center.x, 2) + 
            Math.pow(currentView.eye.y - center.y, 2) + 
            Math.pow(currentView.eye.z - center.z, 2)
        );
        
        var angle = 0;
        rotationInterval = setInterval(function() {
            angle += 0.02; // Rotation speed
            
            var newEye = {
                x: center.x + radius * Math.cos(angle) * 0.7,
                y: center.y + radius * Math.sin(angle) * 0.7,
                z: center.z + radius * 0.5
            };
            
            var update = {
                'scene.camera.eye': newEye
            };
            
            Plotly.relayout(plotlyDiv, update);
        }, 50);
    }
    
    // Set default view after page loading is complete
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            // Use Fit All as default view, no button operation required
            var plotlyDiv = document.querySelector('.plotly-graph-div');
            if (plotlyDiv && views.fit_all) {
                var update = {
                    'scene.camera': views.fit_all
                };
                Plotly.relayout(plotlyDiv, update);
            }
        }, 1000);
    });
    </script>
    """

    # Add view selector to the beginning of HTML
    return view_selector_js + html_str


def write_html(poses, file, vis_depth=1, xyz_length=0.2, center_size=0.01, xyz_width=2):
    """
    Write camera pose visualization to HTML file with optimized camera view.
    """
    # Calculate basic optimal view parameters
    base_view = compute_optimal_camera_view(poses)

    # Extract trajectory information
    trajectory_center = base_view["trajectory_center"]
    max_range = base_view["max_range"]
    ranges = base_view["ranges"]
    auto_vis_depth = base_view["auto_vis_depth"]
    auto_center_size = base_view["auto_center_size"]

    # Calculate optimal view to see entire trajectory
    # Use larger distance to ensure entire trajectory is visible with better angles
    optimal_distance = (
        max_range * 1.8 * 10
    )  # Increase distance by 10x for better overall view

    # Choose ideal angle that can see the full trajectory
    # Use combination of 45-degree elevation and azimuth for good 3D perspective
    elevation = np.pi / 4  # 45-degree elevation angle
    azimuth = np.pi / 4  # 45-degree azimuth angle

    # Calculate optimal viewing direction
    optimal_direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )

    # Calculate optimal camera position
    camera_eye = trajectory_center + optimal_direction * optimal_distance

    # Verify view coverage - ensure all trajectory points are within reasonable distance
    centers_cam = np.zeros([len(poses), 1, 3])
    centers_world = cam2world(centers_cam, poses)[:, 0]

    # Calculate distances from optimal camera position to all trajectory points
    distances_to_points = np.linalg.norm(centers_world - camera_eye, axis=1)
    max_distance_to_point = np.max(distances_to_points)
    min_distance_to_point = np.min(distances_to_points)

    # If distance variation is too large, the view might not be ideal, adjust accordingly
    if max_distance_to_point / min_distance_to_point > 3.0:
        # Recalculate more balanced distance
        optimal_distance = max_range * 2.2 * 10  # Further increase distance (10x)
        camera_eye = trajectory_center + optimal_direction * optimal_distance

    # Create view dictionary with only optimal view for Auto Rotate
    views = {
        "fit_all": {
            "camera_eye": camera_eye,
            "trajectory_center": trajectory_center,
            "auto_vis_depth": auto_vis_depth,
            "auto_center_size": auto_center_size,
            "max_range": max_range,
            "ranges": ranges,
            "description": "Optimal view to see entire trajectory",
        }
    }

    print(f"Trajectory ranges: x={ranges[0]:.3f}, y={ranges[1]:.3f}, z={ranges[2]:.3f}")
    print(f"Max range: {max_range:.3f}")
    print(f"Auto vis_depth: {auto_vis_depth:.3f}, center_size: {auto_center_size:.3f}")
    print(
        f"Trajectory center: ({trajectory_center[0]:.3f}, {trajectory_center[1]:.3f}, {trajectory_center[2]:.3f})"
    )
    print(
        f"Optimal camera position for full trajectory view: ({camera_eye[0]:.3f}, {camera_eye[1]:.3f}, {camera_eye[2]:.3f})"
    )
    print(f"Camera distance from trajectory center: {optimal_distance:.3f}")
    print(
        f"Distance range to trajectory points: {min_distance_to_point:.3f} - {max_distance_to_point:.3f}"
    )

    xyz_length = xyz_length / 3
    xyz_width = xyz_width
    vis_depth = auto_vis_depth  # Use automatically computed depth
    center_size = auto_center_size  # Use automatically computed size

    traces_poses = plotly_visualize_pose(
        poses,
        vis_depth=vis_depth,
        xyz_length=xyz_length,
        center_size=center_size,
        xyz_width=xyz_width,
        mesh_opacity=0.05,
    )
    traces_all2 = traces_poses
    layout2 = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            dragmode="orbit",
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode="data",
            # Set initial camera view to fully see the trajectory with optimized positioning
            camera=dict(
                eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]),
                center=dict(
                    x=trajectory_center[0],
                    y=trajectory_center[1],
                    z=trajectory_center[2],
                ),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        height=800,
        width=1200,
        showlegend=False,
    )

    fig2 = go.Figure(data=traces_all2, layout=layout2)
    html_str2 = pio.to_html(fig2, full_html=False)

    # Add real-time camera view display functionality
    camera_info_html = """
    <div id="camera-info" style="position: fixed; top: 10px; right: 10px; background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); font-family: monospace; font-size: 12px; z-index: 1000; min-width: 250px;">
        <h4 style="margin: 0 0 10px 0; color: #333;">Camera Info</h4>
        <div><strong>Eye:</strong></div>
        <div>x: <span id="eye-x">2.000</span></div>
        <div>y: <span id="eye-y">2.000</span></div>
        <div>z: <span id="eye-z">1.000</span></div>
        <br>
        <div><strong>Center:</strong></div>
        <div>x: <span id="center-x">0.000</span></div>
        <div>y: <span id="center-y">0.000</span></div>
        <div>z: <span id="center-z">0.000</span></div>
        <br>
        <div><strong>Up:</strong></div>
        <div>x: <span id="up-x">0.000</span></div>
        <div>y: <span id="up-y">0.000</span></div>
        <div>z: <span id="up-z">1.000</span></div>
        <br>
        <button onclick="copyToClipboard()" style="background: #007bff; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; width: 100%;">Copy to Clipboard</button>
    </div>
    
    <script>
    function updateCameraInfo() {
        // Get Plotly chart
        var plotlyDiv = document.querySelector('.plotly-graph-div');
        if (!plotlyDiv) return;
        
        // Listen for camera change events
        plotlyDiv.on('plotly_relayout', function(eventData) {
            if (eventData['scene.camera']) {
                var camera = eventData['scene.camera'];
                updateCameraDisplay(camera);
            }
        });
        
        // Initial display
        setTimeout(function() {
            var gd = plotlyDiv;
            if (gd.layout && gd.layout.scene && gd.layout.scene.camera) {
                updateCameraDisplay(gd.layout.scene.camera);
            }
        }, 1000);
    }
    
    function updateCameraDisplay(camera) {
        if (camera.eye) {
            document.getElementById('eye-x').textContent = camera.eye.x.toFixed(3);
            document.getElementById('eye-y').textContent = camera.eye.y.toFixed(3);
            document.getElementById('eye-z').textContent = camera.eye.z.toFixed(3);
        }
        if (camera.center) {
            document.getElementById('center-x').textContent = camera.center.x.toFixed(3);
            document.getElementById('center-y').textContent = camera.center.y.toFixed(3);
            document.getElementById('center-z').textContent = camera.center.z.toFixed(3);
        }
        if (camera.up) {
            document.getElementById('up-x').textContent = camera.up.x.toFixed(3);
            document.getElementById('up-y').textContent = camera.up.y.toFixed(3);
            document.getElementById('up-z').textContent = camera.up.z.toFixed(3);
        }
    }
    
    function copyToClipboard() {
        var eyeX = document.getElementById('eye-x').textContent;
        var eyeY = document.getElementById('eye-y').textContent;
        var eyeZ = document.getElementById('eye-z').textContent;
        var centerX = document.getElementById('center-x').textContent;
        var centerY = document.getElementById('center-y').textContent;
        var centerZ = document.getElementById('center-z').textContent;
        var upX = document.getElementById('up-x').textContent;
        var upY = document.getElementById('up-y').textContent;
        var upZ = document.getElementById('up-z').textContent;
        
        var cameraConfig = `camera=dict(
    eye=dict(x=${eyeX}, y=${eyeY}, z=${eyeZ}),
    center=dict(x=${centerX}, y=${centerY}, z=${centerZ}),
    up=dict(x=${upX}, y=${upY}, z=${upZ})
)`;
        
        navigator.clipboard.writeText(cameraConfig).then(function() {
            alert('Copy to clipboard successful!');
        }).catch(function(err) {
            console.error('Copy failed:', err);
            // Fallback: Create a temporary textarea
            var textArea = document.createElement('textarea');
            textArea.value = cameraConfig;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            alert('Copy to clipboard successful!');
        });
    }

    // Initialize camera info display
    document.addEventListener('DOMContentLoaded', function() {
        updateCameraInfo();
    });

    // If the page has already loaded
    if (document.readyState === 'complete') {
        updateCameraInfo();
    }
    </script>
    """

    # Add view selector and camera info to HTML
    enhanced_html = add_view_selector_to_html(camera_info_html + html_str2, views)

    file.write(enhanced_html)

    print(f"Enhanced visualized poses are saved to {file.name}")
    # Removed redundant view options printing


def plotly_visualize_pose_animated(
    poses_full,
    vis_depth=0.5,
    xyz_length=0.5,
    center_size=2,
    xyz_width=5,
    mesh_opacity=0.05,
):
    """
    Create plotly visualization traces for camera poses, frame by frame for animation.
    Now shows the full trajectory with future poses as completely transparent.
    """
    N_total = len(poses_full)
    plotly_frames = []

    # Pre-compute data for all poses to ensure consistent layout
    centers_cam = np.zeros([N_total, 1, 3])
    centers_world = cam2world(centers_cam, poses_full)
    centers_world = centers_world[:, 0]
    # Get the camera wireframes for all poses
    vertices, faces, wireframe = get_camera_mesh(poses_full, depth=vis_depth)
    vertices_merged, faces_merged = merge_meshes(vertices, faces)
    wireframe_merged = merge_wireframes_plotly(wireframe)
    # Break up (x,y,z) coordinates.
    wireframe_x, wireframe_y, wireframe_z = unbind_np(wireframe_merged, axis=-1)
    centers_x, centers_y, centers_z = unbind_np(centers_world, axis=-1)
    vertices_x, vertices_y, vertices_z = unbind_np(vertices_merged, axis=-1)

    # Initial frame showing all poses with appropriate transparency
    initial_data = []

    for i in tqdm(range(1, N_total + 1), desc="Generating animation frames"):
        current_frame = i - 1  # Current frame index (0-based)

        # Set the color map for the camera trajectory
        color_map = plt.get_cmap("gist_rainbow")
        center_color = []
        faces_merged_color = []
        wireframe_color = []

        for k in range(N_total):  # Process all poses
            # Set the camera pose colors (with a smooth gradient color map).
            r, g, b, _ = color_map(k / (N_total - 1))
            rgb = np.array([r, g, b]) * 0.8

            # Set transparency based on current frame
            if k < current_frame:  # Past poses - visible with reduced opacity
                # Set transparency based on temporal distance, more distant = more transparent
                time_distance = (current_frame - k) / max(current_frame, 1)
                alpha = 0.15 + 0.25 * (1 - time_distance)  # Transparency range 0.15-0.4
                wireframe_alpha = alpha
                mesh_alpha = alpha * 0.4
            elif k == current_frame:  # Current pose - fully visible
                alpha = 0.8  # Fully opaque, dark display
                wireframe_alpha = 0.8
                mesh_alpha = 0.6
            else:  # Future poses - completely transparent
                alpha = 0.0  # Completely transparent
                wireframe_alpha = 0.0
                mesh_alpha = 0.0

            # Set colors and transparency
            wireframe_color += [np.concatenate([rgb, [wireframe_alpha]])] * 11
            center_color += [np.concatenate([rgb, [alpha]])]
            faces_merged_color += [np.concatenate([rgb, [mesh_alpha]])] * 6

        frame_data = [
            go.Scatter3d(
                x=wireframe_x,
                y=wireframe_y,
                z=wireframe_z,
                mode="lines",
                line=dict(color=wireframe_color, width=1),
            ),
            go.Scatter3d(
                x=centers_x,
                y=centers_y,
                z=centers_z,
                mode="markers",
                marker=dict(color=center_color, size=center_size),
            ),
            go.Mesh3d(
                x=vertices_x,
                y=vertices_y,
                z=vertices_z,
                i=[f[0] for f in faces_merged],
                j=[f[1] for f in faces_merged],
                k=[f[2] for f in faces_merged],
                facecolor=faces_merged_color,
                opacity=0.6,  # Set base opacity for mesh
            ),
        ]

        if i == 1:  # Set initial data for the first frame
            initial_data = frame_data

        plotly_frames.append(go.Frame(data=frame_data, name=str(i)))

    return initial_data, plotly_frames


def write_html_animated(
    poses, file, vis_depth=1, xyz_length=0.2, center_size=0.01, xyz_width=2
):
    """
    Write camera pose visualization with animation to HTML file with optimized camera view.
    """
    # Calculate basic optimal view parameters
    base_view = compute_optimal_camera_view(poses)

    # Extract trajectory information
    trajectory_center = base_view["trajectory_center"]
    max_range = base_view["max_range"]
    ranges = base_view["ranges"]
    auto_vis_depth = base_view["auto_vis_depth"]
    auto_center_size = base_view["auto_center_size"]

    # Calculate optimal view to see entire trajectory
    # Use larger distance to ensure entire trajectory is visible with better angles
    optimal_distance = (
        max_range * 1.8 * 10
    )  # Increase distance by 10x for better overall view

    # Choose ideal angle that can see the full trajectory
    # Use combination of 45-degree elevation and azimuth for good 3D perspective
    elevation = np.pi / 4  # 45-degree elevation angle
    azimuth = np.pi / 4  # 45-degree azimuth angle

    # Calculate optimal viewing direction
    optimal_direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )

    # Calculate optimal camera position
    camera_eye = trajectory_center + optimal_direction * optimal_distance

    # Verify view coverage - ensure all trajectory points are within reasonable distance
    centers_cam = np.zeros([len(poses), 1, 3])
    centers_world = cam2world(centers_cam, poses)[:, 0]

    # Calculate distances from optimal camera position to all trajectory points
    distances_to_points = np.linalg.norm(centers_world - camera_eye, axis=1)
    max_distance_to_point = np.max(distances_to_points)
    min_distance_to_point = np.min(distances_to_points)

    # If distance variation is too large, the view might not be ideal, adjust accordingly
    if max_distance_to_point / min_distance_to_point > 3.0:
        # Recalculate more balanced distance
        optimal_distance = max_range * 2.2 * 10  # Further increase distance (10x)
        camera_eye = trajectory_center + optimal_direction * optimal_distance

    # Adjust parameters for animation
    xyz_length = xyz_length / 3
    xyz_width = xyz_width
    vis_depth = auto_vis_depth  # Use automatically computed depth
    center_size = auto_center_size  # Use automatically computed size

    print(
        f"Animation - Trajectory ranges: x={ranges[0]:.3f}, y={ranges[1]:.3f}, z={ranges[2]:.3f}"
    )
    print(f"Animation - Max range: {max_range:.3f}")
    print(
        f"Animation - Auto vis_depth: {auto_vis_depth:.3f}, center_size: {auto_center_size:.3f}"
    )
    print(
        f"Animation - Trajectory center: ({trajectory_center[0]:.3f}, {trajectory_center[1]:.3f}, {trajectory_center[2]:.3f})"
    )
    print(
        f"Animation - Optimal camera position for full trajectory view: ({camera_eye[0]:.3f}, {camera_eye[1]:.3f}, {camera_eye[2]:.3f})"
    )
    print(f"Animation - Camera distance from trajectory center: {optimal_distance:.3f}")
    print(
        f"Animation - Distance range to trajectory points: {min_distance_to_point:.3f} - {max_distance_to_point:.3f}"
    )

    initial_data, plotly_frames = plotly_visualize_pose_animated(
        poses,
        vis_depth=vis_depth,
        xyz_length=xyz_length,
        center_size=center_size,
        xyz_width=xyz_width,
        mesh_opacity=0.05,
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            dragmode="orbit",
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode="data",
            # Use optimized camera view settings (same 10x distance as write_html)
            camera=dict(
                eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]),
                center=dict(
                    x=trajectory_center[0],
                    y=trajectory_center[1],
                    z=trajectory_center[2],
                ),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        height=800,  # Increased height for better animation display
        width=1200,  # Increased width for better animation display
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    )
                ],
            )
        ],
    )

    fig = go.Figure(data=initial_data, layout=layout, frames=plotly_frames)
    html_str = pio.to_html(fig, full_html=False)
    file.write(html_str)

    print(f"Visualized poses are saved to {file}")


def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    """
    Convert 4-dimensional quaternions to 3x3 rotation matrices.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """

    # Order changed to match scipy format: (i, j, k, r)
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    # Construct rotation matrix elements using quaternion algebra
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),  # R[0,0]
            two_s * (i * j - k * r),  # R[0,1]
            two_s * (i * k + j * r),  # R[0,2]
            two_s * (i * j + k * r),  # R[1,0]
            1 - two_s * (i * i + k * k),  # R[1,1]
            two_s * (j * k - i * r),  # R[1,2]
            two_s * (i * k - j * r),  # R[2,0]
            two_s * (j * k + i * r),  # R[2,1]
            1 - two_s * (i * i + j * j),  # R[2,2]
        ),
        -1,
    )
    return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def pose_from_quaternion(pose):
    """
    Convert quaternion-based pose representation to 4x4 transformation matrices.

    Reference:
        https://github.com/pointrix-project/Geomotion/blob/6ab0c364f1b44ab4ea190085dbf068f62b42727c/geomotion/model/cameras.py#L6
    """
    # Convert numpy array to torch tensor if needed
    if type(pose) == np.ndarray:
        pose = torch.tensor(pose)
    # Add batch dimension if input is 1D
    if len(pose.shape) == 1:
        pose = pose[None]
    # Extract translation and quaternion components
    quat_t = pose[..., :3]  # Translation components [tx, ty, tz]
    quat_r = pose[..., 3:]  # Quaternion components [qi, qj, qk, qr]

    # Initialize world-to-camera transformation matrix
    w2c_matrix = torch.zeros((*list(pose.shape)[:-1], 3, 4), device=pose.device)
    w2c_matrix[..., :3, 3] = quat_t  # Set translation part
    w2c_matrix[..., :3, :3] = quaternion_to_matrix(quat_r)  # Set rotation part
    return w2c_matrix


def viz_poses(i, pth, file, args):
    """
    Visualize camera poses for a sequence and write to HTML file.
    """
    file.write(f"<span style='font-size: 18pt;'>{i} {pth}</span><br>")

    # Load pose data from file
    pose = np.load(pth)

    # Convert quaternion poses to transformation matrices
    poses = pose_from_quaternion(pose)  # Input: (N,7), Output: (N,3,4) w2c matrices
    poses = poses.cpu().numpy()

    # Scale camera positions to reduce distance between camera frustums for better visualization
    scale_factor = getattr(
        args, "scale_factor", 0.3
    )  # Default scale factor 0.3, adjustable via command line parameter

    # Apply scaling to translation part (camera positions) while keeping rotation unchanged
    # Create scaled copy of poses
    poses_scaled = poses.copy()
    poses_scaled[..., :3, 3] = poses[..., :3, 3] * scale_factor

    print(f"Original poses shape: {poses.shape}")
    print(f"Applied scale factor: {scale_factor}")

    # Generate visualization based on dynamic flag
    if args.dynamic:
        write_html_animated(poses_scaled, file, vis_depth=args.vis_depth)
    else:
        write_html(poses_scaled, file, vis_depth=args.vis_depth)


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Visualize camera poses with interactive 3D plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--datas",
        type=str,
        nargs="+",
        required=True,
        help="List of pose file paths (.npy format) to visualize.",
    )
    parser.add_argument(
        "--vis_depth",
        type=float,
        default=0.2,
        help="Depth of camera frustum visualization (default: 0.2).",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=0.3,
        help="Scale factor to reduce distance between cameras - smaller values bring cameras closer together (default: 0.3).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./visualize",
        help="Output directory to save HTML visualization files (default: ./visualize).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Create animated visualization showing camera trajectory progression over time.",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Create output directory and process pose files
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Processing {len(args.datas)} pose file(s)...")
    print(f"Output directory: {args.outdir}")
    print(f"Visualization type: {'Animated' if args.dynamic else 'Static'}")

    with open(f"{args.outdir}/visualize.html", "w") as file:
        for i, pth in enumerate(tqdm(args.datas, desc="Processing pose files")):
            if not os.path.exists(pth):
                print(f"Warning: Path {pth} does not exist, skipping.")
                continue
            print(f"Processing: {pth} (#{i+1})")
            viz_poses(i, pth, file, args)

    print(
        f"Visualization complete! Open {args.outdir}/visualize.html in your browser to view results."
    )
