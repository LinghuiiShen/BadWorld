# attack_first_frame_pgd.py
# -*- coding: utf-8 -*-

import os
os.environ["TQDM_DISABLE"] = "1"

import sys
import math
import json
import types
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from einops import rearrange



from diffsynth import WanVideoAstraPipeline, ModelManager

# 直接复用原本推理脚本里的核心函数
from scripts.infer_demo import (
    replace_dit_model_in_manager,
    add_framepack_components,
    add_moe_components,
    prepare_framepack_sliding_window_with_camera_moe,
)
from diffsynth.models.wan_video_dit_moe import rope_apply

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# ============================================================
# 1. Prompt pool
# ============================================================

DEFAULT_STREET_PROMPTS: List[str] = [
    "A bustling marina with sailboats and a tall ship sits amid urban architecture, bathed in daylight with a surreal, atmospheric glow. The scene presents an aerial view of a bustling marina set against an urban backdrop. Hundreds of sailboats are neatly docked in the harbor, creating a dense pattern of masts and hulls. Adjacent to the marina is a wide boulevard lined with palm trees and urban architecture. A large tall ship is moored on the edge of the marina. The lighting suggests daylight conditions, with an atmospheric perspective that casts a somewhat surreal pallor over the city. The overall tone is one of organized activity and serene maritime leisure.",
    "An animated marina, brimming with sailboats and a majestic tall ship, nestles within an urban landscape, illuminated by the warm glow of daylight. The aerial perspective captures the lively scene of a marina surrounded by city structures.",
    "A vibrant marina filled with sailboats and a grand tall ship is nestled in the heart of the city, aglow with bright daylight. From above, the bustling harbor presents a tapestry of masts and boats against a backdrop of urban buildings.",
    "From a bird's-eye view, a lively marina packed with sailboats and a towering ship contrasts beautifully with the surrounding urban architecture, all highlighted by radiant daylight and a dreamlike quality.",
    "A thriving marina with numerous sailboats and a towering ship is set against the backdrop of an urban environment, basking in the daylight that adds a surreal glow to the scene. The aerial view showcases the orderly arrangement of boats and masts.",
    "An aerial perspective reveals a crowded marina adorned with sailboats and a tall ship, framed by modern buildings, all under the enchanting light of day, creating a serene yet dynamic atmosphere.",
    "Amidst urban structures, a lively marina bustling with sailboats and a prominent tall ship glows in brilliant daylight, forming an aerial view that emphasizes the harmony between city life and maritime leisure.",
    "A tall ship and a multitude of sailboats populate a busy marina surrounded by city architecture, all illuminated by the bright daylight that gives the scene an otherworldly quality from above.",
    "An energetic marina filled with sailboats and a majestic tall ship sits within a cityscape, basking in sunlight that bathes the scene in a surreal and airy light. The aerial view captures the orderly chaos of boats.",
    "The aerial view showcases a vibrant marina, alive with sailboats and a tall ship, framed by urban architecture, all under the luminous daylight that gives the scene a dreamlike quality.",
    "A lively marina, rich with the presence of sailboats and an impressive tall ship, finds itself amid urban architecture, glowing under daylight in a manner that feels almost surreal.",
    "An overhead view captures a bustling marina, abundant with sailboats and a significant tall ship, all surrounded by buildings and palm trees, under the warm daylight that bestows a magical ambiance.",
    "Under the bright daylight, a busy marina filled with sailboats and a tall ship stands amidst urban architecture, creating a picturesque scene that feels both organized and relaxed from above.",
    "The marina, teeming with sailboats and overshadowed by a tall ship, is enveloped by a city, all illuminated by sunlight that casts a soft, surreal glow over the landscape.",
    "An expanse of sailboats and a tall ship fills the marina, which is set against a backdrop of city buildings, all brought to life by the bright daylight and an ethereal atmosphere.",
    "A marina bustling with sailboats and a magnificent tall ship is juxtaposed against an urban skyline, all glowing in daylight that adds an air of surreal tranquility to the scene.",
    "The scene from above presents a busy marina brimming with sailboats, accompanied by a majestic tall ship, surrounded by city architecture and adorned with the soft light of day.",
    "In the heart of the urban landscape, a vibrant marina filled with sailboats and a tall ship shines brightly under the daylight, reflecting a blend of spirited activity and peaceful maritime charm.",
    "An overhead perspective reveals a marina crowded with sailboats and a notable tall ship, enveloped by an urban environment, all under the warm embrace of daylight with a surreal atmosphere.",
    "A busy marina with sleek sailboats and a prominent tall ship is framed by urban architecture, illuminated by the inviting light of day that infuses the scene with a surreal quality.",
    "The lively marina, adorned with numerous sailboats and a towering ship, lies in the center of urban life, bathed in daylight that casts an enchanting glow on the surroundings.",
    "High above the city, a bustling marina is filled with boats and a tall ship, reflecting a lively atmosphere in the daylight that lends a surreal aura to the urban backdrop.",
    "An animated marina dotted with sailboats and a grand tall ship finds its place within a city, basking in the daylight that wraps everything in a dreamy glow.",
    "A tall ship and a myriad of sailboats decorate a lively marina, nestled against urban architecture, all glimmering under the warm, surreal light of day.",
    "The marina, vibrant with sailboats and a striking tall ship, sits among urban buildings, illuminated by daylight that casts a dreamlike quality over the entire scene.",
    "An aerial view captures the organized chaos of a bustling marina filled with sailboats and a tall ship, all set against the backdrop of urban structures and bathed in the golden light of day.",
    "The marina thrives with energy as sailboats and a grand tall ship glisten under daylight, surrounded by urban architecture that enhances the surreal vibe of this bustling scene.",
    "An animated marina filled with sailboats and a majestic tall ship is framed by urban buildings, illuminated by bright daylight that imparts a dreamlike quality to the scene. From above, the marina teems with life as numerous sailboats are docked neatly, their masts creating a striking pattern. Nearby, a spacious boulevard lined with palm trees complements the urban scenery, while a grand tall ship rests at the marina's edge. The daylight casts a luminous atmosphere that adds a surreal touch to the city, combining elements of vibrant activity with peaceful maritime charm.",
    "In the heart of the city, a lively marina buzzes with sailboats and a grand tall ship, all under the bright embrace of daylight that gives the area a unique, ethereal glow. Seen from above, the harbor is a frenzy of neatly arranged boats, their masts forming an intricate layout. Adjacent to the marina, a broad boulevard accompanied by palm trees showcases the urban landscape, while a towering tall ship is docked at the marina's fringe. The lighting creates a spectral atmosphere, blending the urban hustle with a tranquil maritime ambiance.",
    "A vibrant marina, bustling with sailboats and a striking tall ship, is nestled within an urban landscape, all enhanced by the brilliance of daylight that bathes everything in a surreal glow. The aerial perspective reveals a busy harbor filled with orderly rows of sailboats, their masts piercing the sky. Close by, a wide palm-lined boulevard provides a picturesque contrast to the city’s architecture. The impressive tall ship lies moored at the marina, while the daylight sets a dreamlike mood over the scene, merging organized activity with calm nautical leisure.",
    "The lively marina filled with sailboats and a towering tall ship showcases a blend of urban architecture and vibrant activity, all under the soft light of day that adds a surreal touch. Viewed from above, the harbor is a mosaic of masts and hulls, indicating a busy boating life. A spacious boulevard with palm trees runs beside the marina, framing the urban environment beautifully. The tall ship rests along the marina’s perimeter, bathed in light that creates an atmospheric perspective, harmonizing the hustle with a serene maritime vibe."
]


# ============================================================
# 2. Image preprocessing / saving
# ============================================================

def build_image_preprocess():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_condition_image_tensor(image_path: str, device: str) -> torch.Tensor:
    """
    Returns normalized tensor in [-1, 1], shape [1, 3, 480, 832]
    """
    image = Image.open(image_path).convert("RGB")
    image = v2.functional.resize(
        image,
        (480, 832),
        interpolation=v2.InterpolationMode.BILINEAR
    )
    preprocess = build_image_preprocess()
    x = preprocess(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    return x


def save_normalized_tensor_as_image(x: torch.Tensor, save_path: str):
    """
    x: [1,3,H,W] in [-1,1]
    """
    x = x.detach().float().cpu().clamp(-1, 1)
    x = (x * 0.5 + 0.5).clamp(0, 1)
    x = (x[0].permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(x).save(save_path)


# ============================================================
# 3. Proxy video / history sampling
# ============================================================

def sample_history_length() -> int:
    return random.choice([1, 9, 17])

# ============================================================
# 3.1 Trajectory sampling
# ============================================================

def make_bounds_per_frame(args) -> Tuple[np.ndarray, np.ndarray]:
    """
    controls[t] = [yaw_t, forward_t, shift_t]
    """
    low = np.array([-args.yaw_max, 0.0, -args.shift_max], dtype=np.float32)
    high = np.array([args.yaw_max, args.forward_max, args.shift_max], dtype=np.float32)
    low = np.tile(low[None, :], (args.target_frames, 1))
    high = np.tile(high[None, :], (args.target_frames, 1))
    return low, high


def clip_controls_np(controls: np.ndarray, args) -> np.ndarray:
    low, high = make_bounds_per_frame(args)
    return np.clip(controls, low, high)


def smooth_controls_np(controls: np.ndarray, args) -> np.ndarray:
    """
    Applies simple per-step smoothing/clipping to the trajectory to avoid sampling overly jittery paths
    """
    controls = controls.copy()
    if controls.shape[0] <= 1:
        return clip_controls_np(controls, args)

    for t in range(1, controls.shape[0]):
        controls[t, 0] = np.clip(
            controls[t, 0],
            controls[t - 1, 0] - args.delta_yaw_max,
            controls[t - 1, 0] + args.delta_yaw_max,
        )
        controls[t, 1] = np.clip(
            controls[t, 1],
            controls[t - 1, 1] - args.delta_forward_max,
            controls[t - 1, 1] + args.delta_forward_max,
        )
        controls[t, 2] = np.clip(
            controls[t, 2],
            controls[t - 1, 2] - args.delta_shift_max,
            controls[t - 1, 2] + args.delta_shift_max,
        )
    return clip_controls_np(controls, args)


def sample_random_trajectory_controls_np(args) -> np.ndarray:
    """
    Samples a relatively smooth trajectory using a random-walk process
    """
    T = args.target_frames
    controls = np.zeros((T, 3), dtype=np.float32)

    controls[0, 0] = np.random.uniform(-args.yaw_max, args.yaw_max)
    controls[0, 1] = np.random.uniform(0.0, args.forward_max)
    controls[0, 2] = np.random.uniform(-args.shift_max, args.shift_max)

    for t in range(1, T):
        controls[t, 0] = controls[t - 1, 0] + np.random.normal(0.0, args.random_walk_std_yaw)
        controls[t, 1] = controls[t - 1, 1] + np.random.normal(0.0, args.random_walk_std_forward)
        controls[t, 2] = controls[t - 1, 2] + np.random.normal(0.0, args.random_walk_std_shift)

    controls = smooth_controls_np(controls, args)
    return controls


def controls_to_relative_pose_3x4(yaw: float, forward_speed: float, radius_shift: float) -> np.ndarray:
    """
    Constructs the corresponding 3x4 relative pose from the 3 per-frame parameters:
    [ r00 r01 r02 | tx ]
    [ r10 r11 r12 | ty ]
    [ r20 r21 r22 | tz ]
    """
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))

    pose = np.eye(4, dtype=np.float32)
    pose[0, 0] = cos_yaw
    pose[0, 2] = sin_yaw
    pose[2, 0] = -sin_yaw
    pose[2, 2] = cos_yaw
    pose[0, 3] = float(radius_shift)
    pose[2, 3] = -float(forward_speed)
    return pose[:3, :]


def generate_sekai_camera_embeddings_from_controls_sliding(
    target_controls: np.ndarray,
    start_frame: int,
    initial_condition_frames: int,
    new_frames: int,
) -> torch.Tensor:
    """
    Generates the camera embedding from the per-frame 3-parameter trajectory. [M,13]

    First 12 dimensions: flattened 3x4 relative pose
    Last 1 dimension: mask
    """
    assert target_controls.shape == (new_frames, 3), \
        f"Expected target_controls shape {(new_frames, 3)}, got {target_controls.shape}"

    framepack_needed_frames = 1 + 16 + 2 + 1 + new_frames
    max_needed_frames = max(
        start_frame + initial_condition_frames + new_frames,
        framepack_needed_frames,
        30,
    )

    relative_poses: List[torch.Tensor] = []
    for i in range(max_needed_frames):
        if i < initial_condition_frames:
            pose3x4 = np.eye(4, dtype=np.float32)[:3, :]
        elif i < initial_condition_frames + new_frames:
            j = i - initial_condition_frames
            yaw, forward_speed, radius_shift = target_controls[j]
            pose3x4 = controls_to_relative_pose_3x4(
                yaw=float(yaw),
                forward_speed=float(forward_speed),
                radius_shift=float(radius_shift),
            )
        else:
            pose3x4 = np.eye(4, dtype=np.float32)[:3, :]

        relative_poses.append(torch.as_tensor(pose3x4, dtype=torch.float32))

    pose_embedding = torch.stack(relative_poses, dim=0)             # [M,3,4]
    pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')  # [M,12]

    mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
    condition_end = min(start_frame + initial_condition_frames + 1, max_needed_frames)
    mask[start_frame:condition_end] = 1.0

    camera_embedding = torch.cat([pose_embedding, mask], dim=1)     # [M,13]
    return camera_embedding.to(torch.bfloat16)


def format_controls_short(controls: np.ndarray, max_frames: int = 4) -> str:
    rows = []
    for t in range(min(max_frames, controls.shape[0])):
        y, fwd, s = controls[t]
        rows.append(f"[{t}: yaw={y:+.3f}, fwd={fwd:+.3f}, shift={s:+.3f}]")
    if controls.shape[0] > max_frames:
        rows.append("...")
    return " ".join(rows)


# ============================================================
# 4. Model loading (keep same structure as original)
# ============================================================

def load_attack_pipeline(
    dit_path: str,
    wan_model_path: str,
    device: str,
    moe_num_experts: int,
    moe_top_k: int,
    moe_hidden_dim: Optional[int],
) -> Tuple[WanVideoAstraPipeline, torch.dtype]:
    """
    Strictly follows the original infer_demo.py loading logic
    """
    replace_dit_model_in_manager()

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        os.path.join(wan_model_path, "diffusion_pytorch_model.safetensors"),
        os.path.join(wan_model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(wan_model_path, "Wan2.1_VAE.pth"),
    ])
    pipe = WanVideoAstraPipeline.from_model_manager(model_manager, device="cuda")

    # Add traditional camera encoder fallback
    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(13, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))

    # Add FramePack components
    add_framepack_components(pipe.dit)

    # Add MoE components
    moe_config = {
        "num_experts": moe_num_experts,
        "top_k": moe_top_k,
        "hidden_dim": moe_hidden_dim or dim * 2,
        "sekai_input_dim": 13,
        "nuscenes_input_dim": 8,
        "openx_input_dim": 13,
    }
    add_moe_components(pipe.dit, moe_config)

    # Load trained weights
    dit_state_dict = torch.load(dit_path, map_location="cpu")
    pipe.dit.load_state_dict(dit_state_dict, strict=False)
    pipe = pipe.to(device)

    model_dtype = next(pipe.dit.parameters()).dtype
    if hasattr(pipe.dit, "clean_x_embedder"):
        pipe.dit.clean_x_embedder = pipe.dit.clean_x_embedder.to(dtype=model_dtype)

    return pipe, model_dtype


def freeze_all_model_params(pipe: WanVideoAstraPipeline):
    pipe.eval()
    for p in pipe.parameters():
        p.requires_grad_(False)


# ============================================================
# 5. Prompt pool encoding
# ============================================================

def encode_prompt_pool(
    pipe: WanVideoAstraPipeline,
    prompts: List[str],
    device: str,
    dtype: torch.dtype,
) -> List[Dict[str, torch.Tensor]]:
    pool = []
    for p in prompts:
        emb = pipe.encode_prompt(p)
        item = {}
        for k, v in emb.items():
            if torch.is_tensor(v):
                item[k] = v.to(device=device, dtype=dtype)
            else:
                item[k] = v
        pool.append(item)
    return pool


def sample_prompt_emb(prompt_pool: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return random.choice(prompt_pool)


# ============================================================
# 6. One training-step forward
# ============================================================

def crop_latents_center(latents: torch.Tensor, target_h: int = 60, target_w: int = 104) -> torch.Tensor:
    """
    latents: [1,16,T,H,W]
    """
    _, _, _, H, W = latents.shape
    if H > target_h or W > target_w:
        hs = (H - target_h) // 2
        ws = (W - target_w) // 2
        latents = latents[:, :, :, hs:hs + target_h, ws:ws + target_w]
    return latents

def encode_single_frame_to_first_latent(
    pipe: WanVideoAstraPipeline,
    img_tensor: torch.Tensor,
    model_dtype: torch.dtype,
    repeat_T: int = 4,
) -> torch.Tensor:
    """
    Encode only one image into one latent frame, differentiably.

    Args:
        img_tensor: [1,3,480,832], normalized in [-1,1]
    Returns:
        first_latent: [16,1,60,104] (after center crop if needed)
    """
    assert img_tensor.dim() == 4 and img_tensor.shape[0] == 1, \
        f"Expected [1,3,H,W], got {img_tensor.shape}"

    device = img_tensor.device
    frames = img_tensor.unsqueeze(2).repeat(1, 1, repeat_T, 1, 1)   # [1,3,4,H,W]
    frames = frames.to(device=device, dtype=model_dtype)

    latents = pipe.encode_video(
        frames,
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    )

    # pipe.encode_video returns [B,16,T,H,W]
    latents = crop_latents_center(latents, 60, 104)   # [1,16,T,60,104]
    first_latent = latents[0, :, 0:1, :, :]           # [16,1,60,104]
    return first_latent


def build_proxy_history_latents_from_single_frame(
    adv_first_lat: torch.Tensor,
    original_first_lat: torch.Tensor,
    history_length: int,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build proxy history directly in latent space.

    Args:
        adv_first_lat: [16,1,60,104], differentiable
        original_first_lat: [16,1,60,104], cached fixed latent
        history_length: int
    Returns:
        history_latents: [16,T,60,104]
    """
    assert history_length >= 1
    assert adv_first_lat.dim() == 4 and adv_first_lat.shape[1] == 1
    assert original_first_lat.dim() == 4 and original_first_lat.shape[1] == 1

    adv_first_lat = adv_first_lat.to(dtype=model_dtype)
    original_first_lat = original_first_lat.to(dtype=model_dtype)

    if history_length == 1:
        return adv_first_lat

    rest = original_first_lat.repeat(1, history_length - 1, 1, 1)   # [16,T-1,60,104]
    history_latents = torch.cat([adv_first_lat, rest], dim=1)        # [16,T,60,104]
    return history_latents


def run_attack_forward_once(
    pipe: WanVideoAstraPipeline,
    model_dtype: torch.dtype,
    adv_first_frame: torch.Tensor,
    original_first_lat: torch.Tensor,
    prompt_pool: List[Dict[str, torch.Tensor]],
    args,
    num_scheduler_steps: int = 20,
    target_frames: int = 8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    One randomized training step:
      - sample history length
      - sample prompt
      - sample trajectory
      - sample scheduler timestep
      - sample target noise
      - build losses
    """
    device = adv_first_frame.device
    # history_length = 1
    history_length = sample_history_length()
    prompt_emb = sample_prompt_emb(prompt_pool)
    trajectory_controls = sample_random_trajectory_controls_np(args)

    # ----------------------------------------------------
    # Build proxy history in latent space
    # Only the first frame is differentiable / adversarial.
    # Remaining history frames are fixed copies of the benign first-frame latent.
    # ----------------------------------------------------
    adv_first_lat = encode_single_frame_to_first_latent(
        pipe=pipe,
        img_tensor=adv_first_frame,
        model_dtype=model_dtype,
        repeat_T=4,
    )   # [16,1,60,104]

    history_latents = build_proxy_history_latents_from_single_frame(
        adv_first_lat=adv_first_lat,
        original_first_lat=original_first_lat,
        history_length=history_length,
        model_dtype=model_dtype,
    )   # [16,T,60,104]

    C, T, H, W = history_latents.shape

    assert C == 16, f"Expected latent channel=16, got {C}"

    # Sample camera trajectory (sekai only, synthetic)
    camera_embedding_full = generate_sekai_camera_embeddings_from_controls_sliding(
        target_controls=trajectory_controls,
        start_frame=0,
        initial_condition_frames=history_length,
        new_frames=target_frames,
    ).to(device=device, dtype=model_dtype)

    # Prepare framepack inputs
    framepack_data = prepare_framepack_sliding_window_with_camera_moe(
        history_latents=history_latents,
        target_frames_to_generate=target_frames,
        camera_embedding_full=camera_embedding_full,
        modality_type="sekai",
    )

    clean_latents = framepack_data["clean_latents"].unsqueeze(0).to(device=device, dtype=model_dtype)
    clean_latents_2x = framepack_data["clean_latents_2x"].unsqueeze(0).to(device=device, dtype=model_dtype)
    clean_latents_4x = framepack_data["clean_latents_4x"].unsqueeze(0).to(device=device, dtype=model_dtype)
    camera_embedding = framepack_data["camera_embedding"].unsqueeze(0).to(device=device, dtype=model_dtype)

    latent_indices = framepack_data["latent_indices"].unsqueeze(0).cpu()
    clean_latent_indices = framepack_data["clean_latent_indices"].unsqueeze(0).cpu()
    clean_latent_2x_indices = framepack_data["clean_latent_2x_indices"].unsqueeze(0).cpu()
    clean_latent_4x_indices = framepack_data["clean_latent_4x_indices"].unsqueeze(0).cpu()


    # Sample one scheduler timestep and one noise tensor
    pipe.scheduler.set_timesteps(num_scheduler_steps)
    t_idx = random.randint(0, 50)
    timestep = pipe.scheduler.timesteps[t_idx]
    timestep_tensor = timestep.unsqueeze(0).to(device=device, dtype=model_dtype)

    new_latents = torch.randn(
        1, C, target_frames, H, W,
        device=device,
        dtype=model_dtype,
    )

    extra_input = pipe.prepare_extra_input(new_latents)

    pred_v, _ = pipe.dit(
        new_latents,
        timestep=timestep_tensor,
        cam_emb=camera_embedding,
        modality_inputs={"sekai": camera_embedding},
        latent_indices=latent_indices,
        clean_latents=clean_latents,
        clean_latent_indices=clean_latent_indices,
        clean_latents_2x=clean_latents_2x,
        clean_latent_2x_indices=clean_latent_2x_indices,
        clean_latents_4x=clean_latents_4x,
        clean_latent_4x_indices=clean_latent_4x_indices,
        **prompt_emb,
        **extra_input,
    )

    # Loss: minimize pred_v magnitude
    # pred_v shape: [1,16,8,60,104]
    loss_vel = pred_v.float().pow(2).mean()

    aux_info = {
        "history_length": history_length,
        "timestep_index": t_idx,
        "timestep_value": float(timestep.detach().float().cpu().item()),
        "trajectory_summary": format_controls_short(trajectory_controls),
    }

    # return loss_vel, loss_attn, token_meta, aux_info
    return loss_vel, aux_info

# ============================================================
# 7. PGD attack loop
# ============================================================

@torch.no_grad()
def project_and_clamp(
    adv_x: torch.Tensor,
    original_x: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    L_inf projection in normalized image space [-1,1].
    """
    eta = torch.clamp(adv_x - original_x, min=-eps, max=eps)
    adv_x = torch.clamp(original_x + eta, min=-1.0, max=1.0)
    return adv_x


def attack_first_frame_pgd(args):

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "log.txt")
    log_f = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    print("===== Run Args =====")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print("====================")

    # --------------------------------------------------------
    # Load first frame
    # --------------------------------------------------------
    original_first_frame = load_condition_image_tensor(args.input_image, device=device)
    print(f"[Check] normalized first-frame range: "
          f"min={original_first_frame.min().item():.4f}, "
          f"max={original_first_frame.max().item():.4f}")
    print(f"[Check] first-frame shape: {tuple(original_first_frame.shape)}")

    # Adv variable lives in normalized space [-1,1]
    adv_first_frame = original_first_frame.clone().detach()

    # --------------------------------------------------------
    # Load pipeline
    # --------------------------------------------------------
    pipe, model_dtype = load_attack_pipeline(
        dit_path=args.dit_path,
        wan_model_path=args.wan_model_path,
        device=device,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_hidden_dim=args.moe_hidden_dim,
    )
    freeze_all_model_params(pipe)

    # Cache original first-frame latent once (no grad, fixed proxy history source)
    with torch.no_grad():
        original_first_lat = encode_single_frame_to_first_latent(
            pipe=pipe,
            img_tensor=original_first_frame,
            model_dtype=model_dtype,
            repeat_T=4,
        ).detach()

    print(f"[Check] cached original first latent shape: {tuple(original_first_lat.shape)}")
    print(f"[Check] cached original first latent dtype: {original_first_lat.dtype}")

    # --------------------------------------------------------
    # Prompt pool
    # --------------------------------------------------------
    prompt_pool = encode_prompt_pool(
        pipe=pipe,
        prompts=DEFAULT_STREET_PROMPTS[:args.num_prompts],
        device=device,
        dtype=model_dtype,
    )
    print(f"[Info] encoded prompt pool size = {len(prompt_pool)}")


    # --------------------------------------------------------
    # Attack loop
    # --------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    for step in range(1, args.num_steps + 1):
        adv_first_frame = adv_first_frame.detach().clone().requires_grad_(True)

        loss_vel, aux_info = run_attack_forward_once(
            pipe=pipe,
            model_dtype=model_dtype,
            adv_first_frame=adv_first_frame,
            original_first_lat=original_first_lat,
            prompt_pool=prompt_pool,
            args=args,
            num_scheduler_steps=args.num_scheduler_steps,
            target_frames=args.target_frames,
        )

        objective = loss_vel
        objective.backward()

        if adv_first_frame.grad is None:
            raise RuntimeError("adv_first_frame.grad is None. Check gradient flow.")

        with torch.no_grad():
            adv_first_frame = adv_first_frame - args.alpha * adv_first_frame.grad.sign()
            adv_first_frame = project_and_clamp(
                adv_x=adv_first_frame,
                original_x=original_first_frame,
                eps=args.eps,
            )

        obj_val = objective.detach().float().item()
        vel_val = loss_vel.detach().float().item()

        if step % args.log_every == 0 or step == 1:
            print(
                f"[Step {step:04d}] "
                f"vel={vel_val:.6f} | "
                f"T_hist={aux_info['history_length']} | "
                f"t_idx={aux_info['timestep_index']} | "
            )

        if step % args.save_every == 0 or step == args.num_steps:
            cur_path = os.path.join(args.output_dir, f"adv_step_{step:04d}.png")
            save_normalized_tensor_as_image(adv_first_frame, cur_path)
            
    final_path = os.path.join(args.output_dir, "adv_final.png")
    save_normalized_tensor_as_image(adv_first_frame, final_path)
    print(f"[Done] saved final adversarial first frame to: {final_path}")


# ============================================================
# 8. CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser("PGD attack on the input first frame for autoregressive Wan-Astra-FramePack-MoE video generation")
    parser.add_argument("--input_image", type=str, default="./examples/condition_images/garden_1.png")
    parser.add_argument("--dit_path", type=str, default="./models/Astra/checkpoints/diffusion_pytorch_model.ckpt")
    parser.add_argument("--wan_model_path", type=str, default="./models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--output_dir", type=str, default="atk_minV2/garden3")

    parser.add_argument("--device", type=str, default="cuda")

    # attack hyperparams
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--eps", type=float, default=0.05,
                        help="L_inf budget in normalized [-1,1] space.")
    parser.add_argument("--alpha", type=float, default=0.005,
                        help="PGD step size in normalized space.")

    # generation / training randomization
    parser.add_argument("--target_frames", type=int, default=8)
    parser.add_argument("--num_scheduler_steps", type=int, default=1000)
    parser.add_argument("--num_prompts", type=int, default=30)

    # moe config
    parser.add_argument("--moe_num_experts", type=int, default=3)
    parser.add_argument("--moe_top_k", type=int, default=1)
    parser.add_argument("--moe_hidden_dim", type=int, default=None)

    # trajectory bounds: maximum range of the control values for each frame
    parser.add_argument("--yaw_max", type=float, default=0.05)
    parser.add_argument("--forward_max", type=float, default=0.05)
    parser.add_argument("--shift_max", type=float, default=0.025)

    # smoothness / step-change bounds: maximum change between adjacent frames
    parser.add_argument("--delta_yaw_max", type=float, default=0.03)
    parser.add_argument("--delta_forward_max", type=float, default=0.03)
    parser.add_argument("--delta_shift_max", type=float, default=0.015)
    
    # random trajectory sampler: only affects the random-walk strength when randomly sampling trajectories
    parser.add_argument("--random_walk_std_yaw", type=float, default=0.02)
    parser.add_argument("--random_walk_std_forward", type=float, default=0.02)
    parser.add_argument("--random_walk_std_shift", type=float, default=0.01)

    # attention loss compute cost control
    parser.add_argument("--attn_query_chunk", type=int, default=16,
                        help="Chunk size over target queries when computing target->start attention mass. Smaller = safer memory, slower.")

    # logging / saving
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)

    # parser.add_argument("--momentum", type=float, default=0.8)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    attack_first_frame_pgd(args)


if __name__ == "__main__":
    main()                  