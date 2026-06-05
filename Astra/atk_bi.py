# atk_bi.py
# -*- coding: utf-8 -*-

import os
os.environ["TQDM_DISABLE"] = "1"
import matplotlib.pyplot as plt
import sys
import math
import json
import random
import argparse
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2
from einops import rearrange

try:
    import cma  # pip install cma
except Exception:
    cma = None

from diffsynth import WanVideoAstraPipeline, ModelManager

HARD_POOL_HISTORY_LENGTHS = [9, 17]

from scripts.infer_demo import (
    replace_dit_model_in_manager,
    add_framepack_components,
    add_moe_components,
    prepare_framepack_sliding_window_with_camera_moe,
)


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

DEFAULT_STREET_PROMPTS: List[str] = [
    "A quiet, rainy street lined with trees and parked vehicles, featuring a reflective, waterlogged surface and a calm, melancholic atmosphere. It is a rainy day in what appears to be a parking area or a quiet street lined with trees. The ground is wet and reflective from the rain. Several cars are parked along the right side of the road, including a mix of compact cars and small trucks. The road has a crosswalk with yellow and white markings, and a yellow circle indicating a speed limit of 30. The atmosphere is calm and subdued due to the rain, creating a peaceful, slightly melancholic mood.",
    "A serene, rain-soaked street adorned with trees and parked cars, showcasing a glossy surface and an atmosphere filled with quiet reflection. The day is marked by gentle rain, transforming what seems to be a parking area into a tranquil scene. Numerous vehicles, from compact cars to small trucks, are nestled along the right side of the road. A crosswalk, painted in yellow and white, cuts through the wet pavement, accompanied by a yellow circle indicating a speed limit of 30. The rainy weather envelops everything in a calm, slightly wistful ambiance.",
    "A tranquil street, drenched by rain, features trees and parked vehicles, adding to the reflective, puddle-filled ground below. It is a rainy day on what seems to be a quiet road or parking lot, where multiple cars, including compact models and smaller trucks, rest on the right. A crosswalk with yellow and white stripes and a yellow circular speed limit sign of 30 give further detail to the scene. The atmosphere is peaceful yet tinged with a sense of melancholy due to the drizzle.",
    "A still, rainy roadway flanked by trees and parked cars reveals a glistening, water-covered surface that enhances the calm, melancholic vibe. The day is drizzly, inviting a feeling of solitude on this quiet street or parking zone. Various vehicles, ranging from compact automobiles to small trucks, are stationed along the right side of the way. The crosswalk, marked in yellow and white, with a yellow circle displaying a speed limit of 30, completes the picture. Overall, the rain fosters a feeling of serenity mingled with a hint of sadness.",
    "An understated, rainy avenue lined with trees and parked cars presents a shining, wet surface, evoking a peaceful and reflective mood. It’s a rainy day along what looks to be a parking space or a tranquil street. Cars, from small trucks to compact vehicles, are parked on the right-hand side. A crosswalk featuring yellow and white lines, along with a yellow circle notifying a speed limit of 30, adds context. This rain-bathed scene exudes calmude with a hint of melancholy.",
    "A quiet, rain-filled street, bordered by trees and lined with parked vehicles, displays a reflective, puddled ground, along with a soothing and somber mood. The scene captures a rainy day in what could be a serene parking area. Multiple vehicles, including a variety of compact cars and smaller trucks, park along the right side of this pathway. There is a crosswalk, distinguished by its yellow and white paint, and a yellow circular sign indicating a speed limit of 30. The rain evokes a tranquil yet slightly sorrowful feeling.",
    "An idyllic, rain-drenched street, flanked by trees and vehicles at rest, features a reflective surface that enhances the calm and introspective atmosphere. It’s obviously a rainy day on a peaceful street or what resembles a parking lot. Several cars, from compact options to small trucks, sit parked on the right side. A crosswalk marked in yellow and white, with a yellow circle denoting a speed limit of 30, punctuates the scene. The rain adds a layer of tranquility laced with a subtle sense of longing.",
    "A peaceful, rain-streaked street lined with trees and various parked vehicles captures a glossy, reflective ground, creating a calm yet melancholic scene. On this rainy day, the area resembles either a quiet road or a parking zone. A range of cars, from compact models to small trucks, occupy the right side of the street. The crosswalk, adorned with yellow and white stripes along with a speed limit sign of 30 in yellow, marks the path. The rain nurtures an atmosphere of serenity tinged with a hint of sadness.",
    "A beautiful, rainy street, bordered by arches of trees and stationary cars, reveals a shiny, waterlogged surface that contributes to a tranquil, melancholic aura. It’s a rainy day that suggests either a desolate street or a tranquil parking area. Numerous vehicles, including a blend of compact cars and small trucks, are parked to the right. A crosswalk with yellow and white detailing, accompanied by a yellow circle indicating a speed limit of 30, frames the scene. The rain bestows a sense of calm along with a touch of wistfulness.",
    "A lovely, quiet street under rain, populated with trees and parked vehicles, exhibits a reflective, shimmering surface that enhances the calm and contemplative mood. On this rainy day, the scene unfolds as either a peaceful street or a parking area. A variety of vehicles, from compact cars to small trucks, occupy spaces on the right. The crosswalk, marked with yellow and white lines, alongside a yellow speed limit sign of 30, adds detail. The rain creates a tranquil, yet slightly nostalgic atmosphere.",
    "A calm, rain-soaked avenue, lined with trees and parked vehicles, showcases a glimmering, water-covered ground that fosters a peaceful, melancholic ambiance. It’s a rainy day on what appears to be a quiet street or parking area. Several vehicles, including compact cars and small trucks, are stationed neatly along the right side. A crosswalk with its yellow and white markings, combined with a yellow circle signifying a speed limit of 30, highlights the scene. The rain brings a serene yet touchingly somber atmosphere.",
    "An elegant, rainy road, rimmed with trees and parked cars, reveals a reflective, slick surface that adds to the peaceful, introspective mood. The day is damp, making this area seem like a quiet street or parking lot. A variety of vehicles, from small trucks to compact cars, line the right side. The crosswalk, easily spotted with its yellow and white stripes, and a yellow circle showing a speed limit of 30, enhance the scenery. The gentle rain establishes a tranquil feeling with a tinge of nostalgia.",
    "A silent, rain-laden street, flanked by trees and parked vehicles, displays a glossy, waterlogged ground that inspires a calm, reflective atmosphere. On this rainy day, the setting resembles a peaceful street or a quiet parking area. Numerous cars, including compact models and smaller trucks, are parked to the right. The crosswalk, emblazoned in yellow and white, along with a circular yellow speed limit sign of 30, adds clarity to the scene. The rain imparts a soothing, yet slightly melancholic feel.",
    "An exquisite, rainy thoroughfare lined with trees and parked cars unfolds a shiny, waterlogged surface that enhances the serene, contemplative mood. It’s a rainy day, ideally suited for what looks like a quiet street or parking area. Various vehicles, from compact cars to small trucks, are parked on the right. A crosswalk, marked vividly in yellow and white, along with a yellow circle sign indicating a speed limit of 30, details the scene. The rainfall cultivates a calming atmosphere, softly tinged with a sense of loss.",
    "A delicate, rain-soaked street bordered by trees and parked vehicles reveals a shimmering, water-filled ground that evokes a tranquil, reflective atmosphere. This rainy day exudes the charm of a quiet street or parking area. Multiple vehicles, from compact cars to small trucks, line up along the right side. A crosswalk, highlighted in yellow and white, with a yellow circle indicating a speed limit of 30, accentuates the setting. The rain introduces a gentle calmness, laced with a hint of melancholy.",
    "A peaceful street, drenched by rain and lined with trees and parked cars, highlights a glossy, reflective surface, contributing to a calm and melancholic vibe. It appears to be a rainy day in this quiet area, which looks like a parking lot. A mix of compact cars and small trucks rests along the right side of the road. The crosswalk with its yellow and white markings, along with a sign denoting a speed limit of 30, enhances the atmosphere. The rain envelops everything in tranquility, slightly tinged with sadness.",
    "A tranquil, rain-drenched road, bordered by trees and parked vehicles, showcases a reflective, puddle-filled ground that enhances the quiet, contemplative mood. It’s a rainy day here, suggesting a peaceful street or parking area. Numerous cars, including compact models and smaller trucks, are parked along the right edge. A crosswalk, marked in yellow and white, accompanied by a yellow circle sign indicating a speed limit of 30, enriches the scenery. The rain creates a serene ambiance, tinged with a hint of nostalgia.",
    "A still, rainy street lined with trees and parked vehicles displays a shiny, waterlogged surface that fosters a calm and reflective atmosphere. It is a rainy day in what appears to be either a quiet street or a parking area. Varied vehicles, from compact cars to small trucks, are neatly parked along the right side. A crosswalk, marked with distinctive yellow and white lines, accompanied by a yellow circle indicating a speed limit of 30, gives context. The overall rainy mood delivers a sense of tranquility mixed with a touch of melancholy.",
    "An enchanting, rain-soaked street bordered by trees and assorted parked vehicles reveals a reflective, glistening surface, contributing to a serene and melancholic feel. The setting suggests a rainy day along a peaceful street or a quiet parking area. A variety of vehicles—compact cars and small trucks—are stationed to the right. The distinct yellow and white markings of a crosswalk, paired with a yellow speed limit sign of 30, highlight the scene. The gentle rain cultivates a soothing atmosphere mingled with a touch of sadness.",
    "A calm avenue drenched in rain, lined with trees and parked vehicles, showcases a shimmering, reflective surface that creates an atmosphere of contemplation and tranquility. It’s a rainy day that turns this street into a quiet retreat or a subdued parking area. Different vehicles, from compact cars to small trucks, sit parked along the right side. A crosswalk, marked with yellow and white, and a yellow circle indicating a speed limit of 30, enhance the ambiance. The rain adds a layer of peacefulness, tinged with a hint of nostalgia.",
    "A quiet, rain-saturated street adorned with trees and parked vehicles features a glimmering, waterlogged surface that enhances a calm, reflective mood. This rainy day unfolds in what seems to be a serene street or a peaceful parking area. A selection of vehicles, ranging from compact cars to small trucks, rests along the right side. A crosswalk, clearly defined with yellow and white markings, accompanied by a yellow circle illustrating a speed limit of 30, completes the imagery. The rain imbues the atmosphere with tranquility, subtly tinged with melancholy.",
    "A serene street drenched in rain, flanked by trees and parked cars, with a glossy, wet pavement and a tranquil, wistful vibe.",
    "On this rainy day, a peaceful street is adorned with trees and vehicles, showcasing a shiny, water-covered ground that enhances the subdued atmosphere.",
    "An eerie calm pervades a rainy road lined with trees and parked automobiles, where the waterlogged surface reflects the somber mood of the day.",
    "A tranquil street scene captures the essence of a rainy day, complete with trees, parked cars, and a shiny, reflective road that adds to the melancholic ambience.",
    "This rainy street is lined with trees and parked vehicles, displaying a wet pavement that mirrors the overcast sky, creating a soft, melancholic atmosphere.",
    "Amidst the rain, a quiet roadway presents a lineup of vehicles and shade from trees, with a glimmering, water-drenched surface enhancing the pensive mood.",
    "A calm, rainy scene shows a street bordered by trees and parked cars, where puddles create a reflective surface mirroring the day's quiet melancholy.",
    "This street, kissed by rain, boasts trees and parked vehicles, its reflective ground contributing to the overall serene and contemplative ambiance.",
    "A soothing rain graces a quiet road lined with trees and parked cars, where the wet surface and muted colors cultivate a tranquil, somber feel.",
    "The peacefulness of this rainy street, lined with trees and parked vehicles, is amplified by the glistening, water-soaked ground beneath a gray sky."
]




# ============================================================
# 1. Image preprocess / save
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
# 2. Randomization
# ============================================================

def sample_history_length() -> int:
    return random.choice([1, 9, 17])

# ============================================================
# 3. Model loading
# ============================================================

def load_attack_pipeline(
    dit_path: str,
    wan_model_path: str,
    device: str,
    moe_num_experts: int,
    moe_top_k: int,
    moe_hidden_dim: Optional[int],
) -> Tuple[WanVideoAstraPipeline, torch.dtype]:

    replace_dit_model_in_manager()

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        os.path.join(wan_model_path, "diffusion_pytorch_model.safetensors"),
        os.path.join(wan_model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(wan_model_path, "Wan2.1_VAE.pth"),
    ])
    pipe = WanVideoAstraPipeline.from_model_manager(model_manager, device="cuda")

    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(13, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))

    add_framepack_components(pipe.dit)

    moe_config = {
        "num_experts": moe_num_experts,
        "top_k": moe_top_k,
        "hidden_dim": moe_hidden_dim or dim * 2,
        "sekai_input_dim": 13,
        "nuscenes_input_dim": 8,
        "openx_input_dim": 13,
    }
    add_moe_components(pipe.dit, moe_config)

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
# 4. Prompt pool encoding
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
# 5. Latent helpers
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

    device = original_first_lat.device
    adv_first_lat = adv_first_lat.to(device=device, dtype=model_dtype)
    original_first_lat = original_first_lat.to(device=device, dtype=model_dtype)

    if history_length == 1:
        return adv_first_lat

    rest = original_first_lat.repeat(1, history_length - 1, 1, 1)   # [16,T-1,60,104]
    history_latents = torch.cat([adv_first_lat, rest], dim=1)        # [16,T,60,104]
    return history_latents


# ============================================================
# 6. Trajectory parameterization: per-frame (yaw, forward, shift)
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

def flatten_controls_np(controls: np.ndarray) -> np.ndarray:
    return controls.astype(np.float32).reshape(-1)


def unflatten_controls_np(flat: np.ndarray, target_frames: int) -> np.ndarray:
    return np.asarray(flat, dtype=np.float32).reshape(target_frames, 3)


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



def sample_random_trajectory_controls_np(args):
    T = args.target_frames
    controls = np.zeros((T, 3), dtype=np.float32)

    # first frame initialization, same as the first script
    controls[0, 0] = np.random.uniform(-args.yaw_max, args.yaw_max)
    controls[0, 1] = np.random.uniform(0.0, args.forward_max)
    controls[0, 2] = np.random.uniform(-args.shift_max, args.shift_max)

    # subsequent frames: random walk
    for t in range(1, T):
        controls[t, 0] = controls[t - 1, 0] + np.random.normal(0, args.random_walk_std_yaw)
        controls[t, 1] = controls[t - 1, 1] + np.random.normal(0, args.random_walk_std_forward)
        controls[t, 2] = controls[t - 1, 2] + np.random.normal(0, args.random_walk_std_shift)

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

    pose_embedding = torch.stack(relative_poses, dim=0)          # [M,3,4]
    pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')  # [M,12]

    mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
    condition_end = min(start_frame + initial_condition_frames + 1, max_needed_frames)
    mask[start_frame:condition_end] = 1.0

    camera_embedding = torch.cat([pose_embedding, mask], dim=1)  # [M,13]
    return camera_embedding.to(torch.bfloat16)


def format_controls_short(controls: np.ndarray, max_frames: Optional[int] = None) -> str: #打印全，实验
    rows = []
    n = controls.shape[0] if max_frames is None else min(max_frames, controls.shape[0])

    for t in range(n):
        y, fwd, s = controls[t]
        rows.append(f"[{t}: yaw={y:+.3f}, fwd={fwd:+.3f}, shift={s:+.3f}]")

    if max_frames is not None and controls.shape[0] > max_frames:
        rows.append("...")

    return " ".join(rows)


# ============================================================
# 7. One forward with a fixed trajectory
# ============================================================

def run_attack_forward_once(
    pipe: WanVideoAstraPipeline,
    model_dtype: torch.dtype,
    adv_first_frame: torch.Tensor,
    original_first_lat: torch.Tensor,
    prompt_pool: List[Dict[str, torch.Tensor]],
    trajectory_controls: np.ndarray, 
    num_scheduler_steps: int = 20,
    target_frames: int = 8,
    max_timestep_index: int = 50,
    cached_adv_first_lat: Optional[torch.Tensor] = None,
    fixed_eval_context: Optional[Dict[str, Any]] = None,
    fixed_history_length: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    device = adv_first_frame.device

    if fixed_eval_context is None:
        history_length = fixed_history_length if fixed_history_length is not None else sample_history_length()
        prompt_emb = sample_prompt_emb(prompt_pool)
    else:
        history_length = fixed_eval_context["history_length"]
        prompt_emb = prompt_pool[fixed_eval_context["prompt_idx"]]

    # differentiable first-frame latent
    if cached_adv_first_lat is None:  # adv first latent remains unchanged every cma
        adv_first_lat = encode_single_frame_to_first_latent(
            pipe=pipe,
            img_tensor=adv_first_frame,
            model_dtype=model_dtype,
            repeat_T=4,
        )
    else:
        adv_first_lat = cached_adv_first_lat.to(device=device, dtype=model_dtype)

    # proxy history
    history_latents = build_proxy_history_latents_from_single_frame(
        adv_first_lat=adv_first_lat,
        original_first_lat=original_first_lat,
        history_length=history_length,
        model_dtype=model_dtype,
    )   # [16,T,60,104]

    C, T, H, W = history_latents.shape
    assert C == 16, f"Expected latent channel=16, got {C}"

    # build camera embedding from fixed controls
    camera_embedding_full = generate_sekai_camera_embeddings_from_controls_sliding(
        target_controls=trajectory_controls,
        start_frame=0,
        initial_condition_frames=history_length,
        new_frames=target_frames,
    ).to(device=device, dtype=model_dtype)

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


    pipe.scheduler.set_timesteps(num_scheduler_steps)
    max_valid_t = min(max_timestep_index, len(pipe.scheduler.timesteps) - 1)

    if fixed_eval_context is None:
        t_idx = random.randint(0, max_valid_t)
    else:
        t_idx = min(int(fixed_eval_context["timestep_index"]), max_valid_t)

    timestep = pipe.scheduler.timesteps[t_idx]
    timestep_tensor = timestep.unsqueeze(0).to(device=device, dtype=model_dtype)


    if fixed_eval_context is None:
        new_latents = torch.randn(
            1, C, target_frames, H, W,
            device=device,
            dtype=model_dtype,
        )
    else:
        new_latents = fixed_eval_context["new_latents"].to(device=device, dtype=model_dtype)

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

    loss_vel = pred_v.float().pow(2).mean()

    aux_info = {
        "history_length": history_length,
        "timestep_index": t_idx,
        "timestep_value": float(timestep.detach().float().cpu().item()),
        "trajectory_summary": format_controls_short(trajectory_controls),
    }
    return loss_vel, aux_info


# ============================================================
# 8. Projection
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


# ============================================================
# 9. Inner attack step
# ============================================================

def inner_attack_step(
    pipe: WanVideoAstraPipeline,
    model_dtype: torch.dtype,
    adv_first_frame: torch.Tensor,
    original_first_frame: torch.Tensor,
    original_first_lat: torch.Tensor,
    prompt_pool: List[Dict[str, torch.Tensor]],
    trajectory_controls: np.ndarray,
    args,
    fixed_history_length: Optional[int] = None
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:

    adv_first_frame = adv_first_frame.detach().clone().requires_grad_(True)

    loss_vel, aux_info = run_attack_forward_once(
        pipe=pipe,
        model_dtype=model_dtype,
        adv_first_frame=adv_first_frame,
        original_first_lat=original_first_lat,
        prompt_pool=prompt_pool,
        trajectory_controls=trajectory_controls,
        num_scheduler_steps=args.num_scheduler_steps,
        target_frames=args.target_frames,
        max_timestep_index=args.max_timestep_index,
        fixed_history_length=fixed_history_length,
    )

    objective = loss_vel
    objective.backward()

    if adv_first_frame.grad is None:
        raise RuntimeError("adv_first_frame.grad is None. Check gradient flow.")

    with torch.no_grad():
        # minimize pred_v norm
        adv_first_frame = adv_first_frame - args.alpha * adv_first_frame.grad.sign()

        adv_first_frame = project_and_clamp(
            adv_x=adv_first_frame,
            original_x=original_first_frame,
            eps=args.eps,
        )

    return adv_first_frame, float(objective.detach().float().item()), aux_info

# ============================================================
# 10. Outer fitness evaluation
# ============================================================

@torch.no_grad()
def evaluate_trajectory_fitness(
    pipe: WanVideoAstraPipeline,
    model_dtype: torch.dtype,
    adv_first_frame: torch.Tensor,
    original_first_lat: torch.Tensor,
    prompt_pool: List[Dict[str, torch.Tensor]],
    trajectory_controls: np.ndarray,
    args,
    mc_samples: int,
    cached_adv_first_lat: Optional[torch.Tensor] = None,
    fixed_eval_contexts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:

    raw_scores: List[float] = []
    for i in range(mc_samples):
        ctx = None if fixed_eval_contexts is None else fixed_eval_contexts[i]

        loss_vel, _ = run_attack_forward_once(
            pipe=pipe,
            model_dtype=model_dtype,
            adv_first_frame=adv_first_frame,
            original_first_lat=original_first_lat,
            prompt_pool=prompt_pool,
            trajectory_controls=trajectory_controls,
            num_scheduler_steps=args.num_scheduler_steps,
            target_frames=args.target_frames,
            max_timestep_index=args.max_timestep_index,
            cached_adv_first_lat=cached_adv_first_lat,
            fixed_eval_context=ctx,
        )

        raw_scores.append(float(loss_vel.detach().float().item()))

    mean_raw = float(np.mean(raw_scores))
    fitness = mean_raw

    return {
        "fitness": fitness,
        "raw_mean": mean_raw,
        "controls": trajectory_controls.astype(np.float32).copy(),
        "controls_flat": flatten_controls_np(trajectory_controls),
        "mc_scores": raw_scores,
    }


def deduplicate_pool_entries(entries: List[Dict[str, Any]], target_frames: int) -> List[Dict[str, Any]]:
    seen = set()
    kept = []
    for item in entries:
        flat = tuple(np.round(item["controls"].reshape(target_frames * 3), 6).tolist())
        if flat in seen:
            continue
        seen.add(flat)
        kept.append(item)
    return kept


# ============================================================
# 11. Outer search: CMA-ES hard trajectory pool
# ============================================================
def search_hard_trajectory_pool_cma(
    pipe: WanVideoAstraPipeline,
    model_dtype: torch.dtype,
    adv_first_frame: torch.Tensor,
    original_first_lat: torch.Tensor,
    prompt_pool: List[Dict[str, torch.Tensor]],
    args,
    fixed_history_length: int,
    vis_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Output the top pool_size entries of the hardest trajectory pool (sorted by fitness)
    High fitness : pred_v norm remains large under the current adv image --> harder
    """
    all_evals: List[Dict[str, Any]] = []
    cma_snapshots: List[List[Dict[str, Any]]] = []

    with torch.no_grad():
        cached_adv_first_lat = encode_single_frame_to_first_latent(
            pipe=pipe,
            img_tensor=adv_first_frame,
            model_dtype=model_dtype,
            repeat_T=4,
        ).detach()

        device = adv_first_frame.device


        pipe.scheduler.set_timesteps(args.num_scheduler_steps)

        num_timesteps = len(pipe.scheduler.timesteps)

        outer_min_t_idx = max(0, int(args.outer_min_timestep_index))
        outer_max_t_idx = min(int(args.outer_max_timestep_index), num_timesteps - 1)

        if outer_min_t_idx > outer_max_t_idx:
            raise ValueError(
                f"Invalid outer timestep range: "
                f"outer_min_timestep_index={outer_min_t_idx}, "
                f"outer_max_timestep_index={outer_max_t_idx}, "
                f"num_timesteps={num_timesteps}"
            )

        used_ctx_keys = set()
        fixed_eval_contexts = []

        for _ in range(args.outer_mc_samples):
            while True:
                history_length = fixed_history_length

                prompt_idx = 0

                # Outer loop uses larger indices: lower-noise / middle denoising region.
                timestep_index = random.randint(outer_min_t_idx, outer_max_t_idx)

                key = (history_length, prompt_idx, timestep_index)

                if key not in used_ctx_keys:
                    used_ctx_keys.add(key)
                    break

            timestep = pipe.scheduler.timesteps[timestep_index]

            fixed_eval_contexts.append({
                "history_length": history_length,
                "prompt_idx": prompt_idx,
                "timestep_index": timestep_index,
                "new_latents": torch.randn(
                    1, 16, args.target_frames, 60, 104,
                    device=device,
                    dtype=model_dtype,
                ),
            })

        print("[Outer Eval Contexts]")
        for i, ctx in enumerate(fixed_eval_contexts):
            print(
                f"  mc={i} | " #The i-th random sample
                f"T_hist={ctx['history_length']} | "
                f"prompt_idx={ctx['prompt_idx']} | "
                f"t_idx={ctx['timestep_index']}"
            )

    lower, upper = make_bounds_per_frame(args)
    lower_flat = flatten_controls_np(lower)
    upper_flat = flatten_controls_np(upper)

    def score_flat(x_flat: np.ndarray) -> float:
        controls = unflatten_controls_np(np.asarray(x_flat, dtype=np.float32), args.target_frames) # Reshape the one-dimensional parameter vector into [T, 3]
        controls = smooth_controls_np(controls, args)

        result = evaluate_trajectory_fitness(
            pipe=pipe,
            model_dtype=model_dtype,
            adv_first_frame=adv_first_frame,
            original_first_lat=original_first_lat,
            prompt_pool=prompt_pool,
            trajectory_controls=controls,
            args=args,
            mc_samples=args.outer_mc_samples,
            cached_adv_first_lat=cached_adv_first_lat,
            fixed_eval_contexts=fixed_eval_contexts,
        )
        all_evals.append(result)
        return result["fitness"]

#####################

    if cma is not None:

        x0_controls = np.zeros((args.target_frames, 3), dtype=np.float32)

        x0_controls[:, 0] = 0.0   # yaw center
        x0_controls[:, 1] = 0.025 # forward center of [0, 0.05]
        x0_controls[:, 2] = 0.0   # shift center

        x0 = flatten_controls_np(x0_controls)

        x0 = np.clip(x0, lower_flat, upper_flat)

        sigma0 = args.cma_sigma
        opts = {
            "bounds": [lower_flat.tolist(), upper_flat.tolist()],
            "popsize": args.cma_popsize, 
            "verbose": -9,
            "verb_log": 0,
            "verb_disp": 0,
            "maxiter": args.cma_iters, #8
            "seed": int(args.seed + 1000)
        }
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts) #######
# CMA searches over a 24-dimensional vector: x = [yaw_0, fwd_0, shift_0, ..., yaw_7, fwd_7, shift_7]
        while not es.stop():
            xs = es.ask() # sample a batch of candidate solutions from the current Gaussian distribution
            vals = []
            for x in xs: 
                fitness = score_flat(np.asarray(x, dtype=np.float32)) # compute fitness for each candidate
                vals.append(-fitness)   # CMA minimizes by default, so use the negative value
            es.tell(xs, vals) # update the distribution
            # Record all accumulated candidates after the current CMA iteration finishes
            cma_snapshots.append(list(all_evals))

            if es.countiter >= args.cma_iters:
                break
                
            # Feed this batch of candidate solutions and their objective values back to CMA to update the search distribution:
            #     update the mean
            #     update the covariance
            #     update the step size
            # Then the next ask() will be biased toward trajectory regions currently considered "harder".   

    else:
        print("[Warn] `cma` package not found. Falling back to random search for hard trajectory pool.")
        for _ in range(args.random_pool_candidates):
            score_flat(flatten_controls_np(sample_random_trajectory_controls_np(args)))


    if len(all_evals) == 0:
        raise RuntimeError("Hard trajectory search produced zero evaluated candidates.")

    all_evals = deduplicate_pool_entries(all_evals, args.target_frames)
    all_evals = sorted(all_evals, key=lambda x: x["fitness"], reverse=True) #sort
    pool = all_evals[:args.pool_size] #select top

    print(
        f"[Outer] searched {len(all_evals)} unique trajectories | "
    )
    return pool


def sample_trajectory_from_pool(
    pool: List[Dict[str, Any]],
    mode: str = "weighted",
    temperature: float = 1.0,
) -> np.ndarray:
    assert len(pool) > 0, "trajectory pool is empty"

    if mode == "uniform":
        item = random.choice(pool)
        return item["controls"].copy()

    scores = np.array([p["fitness"] for p in pool], dtype=np.float64)
    scores = scores - scores.max()
    probs = np.exp(scores / max(temperature, 1e-6))
    probs = probs / probs.sum()
    idx = np.random.choice(len(pool), p=probs)
    return pool[int(idx)]["controls"].copy()


def save_pool_json(pool: List[Dict[str, Any]], save_path: str):
    payload = []
    for idx, item in enumerate(pool):
        payload.append({
            "rank": idx,
            "fitness": float(item["fitness"]),
            "raw_mean": float(item["raw_mean"]),
            "controls": np.asarray(item["controls"]).tolist(),
        })
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

# ============================================================
# 12. Main bilevel attack loop
# ============================================================

def attack_first_frame_bilevel(args):
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
    print(f"[Check] normalized first-frame range: min={original_first_frame.min().item():.4f}, max={original_first_frame.max().item():.4f}")
    print(f"[Check] first-frame shape: {tuple(original_first_frame.shape)}")

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

    # cache benign first latent
    with torch.no_grad():
        original_first_lat = encode_single_frame_to_first_latent(
            pipe=pipe,
            img_tensor=original_first_frame,
            model_dtype=model_dtype,
            repeat_T=4,
        ).detach()

    print(f"[Check] cached original first latent shape: {tuple(original_first_lat.shape)}")
    print(f"[Check] cached original first latent dtype: {original_first_lat.dtype}")

    # prompt pool
    prompt_pool = encode_prompt_pool(
        pipe=pipe,
        prompts=DEFAULT_STREET_PROMPTS[:args.num_prompts],
        device=device,
        dtype=model_dtype,
    )
    print(f"[Info] encoded prompt pool size = {len(prompt_pool)}")

    # --------------------------------------------------------
    # Attack loop
    # -------------------------------------------------------

    global_step = 0
    round_idx = 0
    hard_pools: Dict[int, List[Dict[str, Any]]] = {}

    # ===== Phase A: warmup ===== 
    warmup_steps = min(args.warmup_steps, args.num_steps)
    print(f"[Phase] warmup inner attack for {warmup_steps} steps using random trajectories")
    for _ in range(warmup_steps):
        controls = sample_random_trajectory_controls_np(args)

        adv_first_frame, obj_val, aux_info = inner_attack_step(
            pipe=pipe,
            model_dtype=model_dtype,
            adv_first_frame=adv_first_frame,
            original_first_frame=original_first_frame,
            original_first_lat=original_first_lat,
            prompt_pool=prompt_pool,
            trajectory_controls=controls,
            args=args,
        )
        global_step += 1

        if global_step % args.log_every == 0 or global_step == 1:
            print(
                f"[Warmup {global_step:04d}] obj={obj_val:.6f} | "
                f"T_hist={aux_info['history_length']} | "
                f"t_idx={aux_info['timestep_index']} | "
                # f"traj={aux_info['trajectory_summary']}"
            )

    # ===== Phase B: alternate outer search and inner update =====
    while global_step < args.num_steps:
        round_idx += 1
        print(f"\n========== Outer Round {round_idx:03d} ==========")
        args.current_round_idx = round_idx
        for hist_len in HARD_POOL_HISTORY_LENGTHS:
            vis_dir = os.path.join(
                args.output_dir,
                f"r{round_idx:03d}",
                f"h{hist_len:02d}"
            )
            os.makedirs(vis_dir, exist_ok=True)

            print(f"\n[Outer] searching hard pool for history_length={hist_len}")

            hard_pools[hist_len] = search_hard_trajectory_pool_cma(
                pipe=pipe,
                model_dtype=model_dtype,
                adv_first_frame=adv_first_frame.detach(),
                original_first_lat=original_first_lat,
                prompt_pool=prompt_pool,
                args=args,
                fixed_history_length=hist_len,
                vis_dir=vis_dir,
            )

            pool_json_path = os.path.join(
                args.output_dir,
                f"hard_pool_round_{round_idx:03d}_hist_{hist_len:02d}.json"
            )
            save_pool_json(hard_pools[hist_len], pool_json_path)
            print(f"[Outer] saved hard trajectory pool to: {pool_json_path}")

        inner_steps_this_round = min(args.inner_steps_per_round, args.num_steps - global_step)
        print(f"[Phase] inner attack for {inner_steps_this_round} steps using current hard pool")

        for _ in range(inner_steps_this_round):

            hist_len = sample_history_length()

            if hist_len == 1:
                controls = sample_random_trajectory_controls_np(args)
            else:
                pool_for_hist = hard_pools[hist_len]

                controls = sample_trajectory_from_pool(
                    pool_for_hist,
                    mode=args.pool_sample_mode,
                    temperature=args.pool_sample_temperature,
                )

            adv_first_frame, obj_val, aux_info = inner_attack_step(
                pipe=pipe,
                model_dtype=model_dtype,
                adv_first_frame=adv_first_frame,
                original_first_frame=original_first_frame,
                original_first_lat=original_first_lat,
                prompt_pool=prompt_pool,
                trajectory_controls=controls,
                args=args,
                fixed_history_length=hist_len,
            )
            global_step += 1


            if global_step % args.log_every == 0 or global_step == 1:
                print(
                    f"[Step {global_step:04d}] obj={obj_val:.6f} | "
                    f"T_hist={aux_info['history_length']} | "
                    f"t_idx={aux_info['timestep_index']} | "
                )

            if global_step % args.save_every == 0 or global_step == args.num_steps:
                cur_path = os.path.join(args.output_dir, f"adv_step_{global_step:04d}.png")
                save_normalized_tensor_as_image(adv_first_frame, cur_path)
                print(f"[Save] current adv -> {cur_path}")

    final_path = os.path.join(args.output_dir, "adv_final.png")
    save_normalized_tensor_as_image(adv_first_frame, final_path)
    print(f"[Done] saved final adversarial first frame to: {final_path}")


# ============================================================
# 13. CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser("Bilevel PGD attack with CMA-ES hard-trajectory search for Wan-Astra-FramePack-MoE")

    parser.add_argument("--input_image", type=str, default="./datasets/imgs_processed/3.jpg")
    parser.add_argument("--dit_path", type=str, default="./models/Astra/checkpoints/diffusion_pytorch_model.ckpt")
    parser.add_argument("--wan_model_path", type=str, default="./models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--output_dir", type=str, default="atk_bi/3")

    parser.add_argument("--device", type=str, default="cuda")

    # image attack
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--eps", type=float, default=0.05, help="L_inf budget in normalized [-1,1] space.")
    parser.add_argument("--alpha", type=float, default=0.005, help="PGD step size in normalized space.")

    # generation randomization
    parser.add_argument("--target_frames", type=int, default=8)
    parser.add_argument("--num_scheduler_steps", type=int, default=1000)
    parser.add_argument("--max_timestep_index", type=int, default=50)
    parser.add_argument("--num_prompts", type=int, default=8)

    # MoE config
    parser.add_argument("--moe_num_experts", type=int, default=3)
    parser.add_argument("--moe_top_k", type=int, default=1)
    parser.add_argument("--moe_hidden_dim", type=int, default=None)

    # bilevel schedule
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--inner_steps_per_round", type=int, default=50)

    # hard pool
    parser.add_argument("--pool_size", type=int, default=30)
    parser.add_argument("--pool_sample_mode", type=str, default="uniform", choices=["uniform", "weighted"])
    parser.add_argument("--pool_sample_temperature", type=float, default=1.0)
                                    #not used. only used in weighted mode。

    # outer search
    parser.add_argument("--outer_mc_samples", type=int, default=1) #每条 candidate trajectory 评估几次，然后取平均 fitness
    parser.add_argument("--cma_iters", type=int, default=4) #CMA（或 fallback 搜索）做多少轮迭代
    parser.add_argument("--cma_popsize", type=int, default=20) #每轮 CMA 采样多少条 candidate trajectory
    parser.add_argument("--cma_sigma", type=float, default=0.02) #CMA 初始采样步长
    parser.add_argument("--random_pool_candidates", type=int, default=200) #只有 没装 cma 包 时才用。

    parser.add_argument(
        "--outer_min_timestep_index",
        type=int,
        default=0,
        help="Outer-loop trajectory search uses larger timestep indices, i.e. lower-noise / middle denoising region."
    )

    parser.add_argument(
        "--outer_max_timestep_index",
        type=int,
        default=50,
        help="Maximum timestep index used by outer-loop trajectory search."
    )

    # trajectory bounds 每一帧控制量的最大范围
    parser.add_argument("--yaw_max", type=float, default=0.05)
    parser.add_argument("--forward_max", type=float, default=0.05)
    parser.add_argument("--shift_max", type=float, default=0.025)

    # smoothness / step-change bounds 相邻两帧之间最多变化
    parser.add_argument("--delta_yaw_max", type=float, default=0.03)
    parser.add_argument("--delta_forward_max", type=float, default=0.03)
    parser.add_argument("--delta_shift_max", type=float, default=0.015)

    # random trajectory sampler 只影响 随机采样 trajectory 时的 random walk 强度
    # parser.add_argument("--random_walk_std_yaw", type=float, default=0.03)
    # parser.add_argument("--random_walk_std_forward", type=float, default=0.03)
    # parser.add_argument("--random_awalk_std_shift", type=float, default=0.015)

    # parser.add_argument("--random_walk_std_yaw", type=float, default=0.015)
    # parser.add_argument("--random_walk_std_forward", type=float, default=0.015)
    # parser.add_argument("--random_walk_std_shift", type=float, default=0.008)

    parser.add_argument("--random_walk_std_yaw", type=float, default=0.02)
    parser.add_argument("--random_walk_std_forward", type=float, default=0.02)
    parser.add_argument("--random_walk_std_shift", type=float, default=0.01)

    # logging / saving
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    attack_first_frame_bilevel(args)


if __name__ == "__main__":
    main()