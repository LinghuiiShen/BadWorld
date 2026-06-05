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
    generate_sekai_camera_embeddings_sliding,
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
    "A sunlit European street lined with historic buildings and vibrant greenery creates a warm, charming, and inviting atmosphere. The scene shows a picturesque open square paved with red bricks, surrounded by classic narrow townhouses featuring tall windows, gabled roofs, and dark-painted facades. On the right side, a lush arrangement of potted plants and blooming flowers adds rich color and texture to the foreground. A vintage-style streetlamp stands prominently near the center-right, contributing to the timeless character of the street. Mature trees frame the background, their leaves glowing in the warm afternoon sunlight. Bicycles rest along the edges of the buildings, reinforcing the urban yet leisurely feel. The sky is bright blue with scattered clouds, and soft sun flares enter from the left.",
    "A peaceful old-town street stretches forward beneath a soft afternoon sky, surrounded by elegant brick buildings, leafy trees, and decorative storefronts. Warm sunlight washes across the cobblestone pavement, revealing subtle texture and long shadows. Flower boxes spill from windows, while bicycles lean quietly against stone walls and iron railings. A small cafe terrace appears on one side with neatly arranged chairs and potted plants, suggesting a calm and inviting public space. The facades of the surrounding buildings vary in height and color, giving the street a layered visual rhythm. The entire setting feels quiet, cinematic, and richly detailed, with a balance of architecture, greenery, and open urban space.",
    "A bright residential street in a charming European neighborhood is framed by narrow townhouses, mature trees, and a clean red-brick roadway. The architecture features tall windows, steep roofs, and decorative trim that create a refined historic atmosphere. Soft sunlight reflects off the upper floors and filters through the leaves, casting dappled shadows on the pavement below. Along the sidewalk, potted flowers and shrubs add bursts of color and texture. A vintage lamppost stands beside a bench near the edge of a small square, giving the scene a timeless public-street character. The air feels fresh, the sky is lightly clouded, and the overall mood is calm, open, and welcoming.",
    "A quiet urban street opens into a small square bordered by elegant buildings with dark facades, tall windows, and classic rooflines. The brick pavement glows warmly under late afternoon sunlight, while trees and climbing plants soften the architectural edges. A cluster of flowers in decorative pots occupies the foreground, bringing vivid natural detail close to the viewer. Several bicycles are parked casually along the buildings, reinforcing the everyday character of the space. In the distance, more houses and greenery create depth and a sense of continuity. The light is gentle and golden, the sky is bright, and the whole scene feels like a tranquil, photorealistic European street environment.",
    "A picturesque pedestrian-friendly street is illuminated by warm sunlight and framed by old brick houses, green foliage, and ornamental street elements. The road surface is composed of rich reddish paving stones, leading the eye toward a softly lit urban square. On one side, flowerpots overflow with blossoms and leafy plants, while on the other side, storefront windows and doors create a rhythm of vertical details. Mature trees rise behind the buildings, partially filtering the sun and adding layered shadows. A classic streetlamp and a few parked bicycles contribute to the lived-in atmosphere. The scene feels peaceful, detailed, and ideal for cinematic city-view generation with strong architectural identity.",
    "A narrow European street bathed in gentle sunshine features historic buildings, colorful plants, and carefully textured paving stones. The composition emphasizes depth, with the street receding toward a bright open square framed by trees and rooftops. The buildings have tall proportions, dark-painted facades, and old-world roof shapes that evoke a timeless urban neighborhood. On the right side, decorative potted plants and flowers fill the foreground with life and color. The left side receives a soft burst of sunlight that creates natural glow and subtle flare. A few bicycles and a traditional lamp anchor the everyday realism of the space, producing a calm, inviting, and richly atmospheric street scene.",
    "A beautiful town street unfolds between classic narrow houses with tall windows, sloped roofs, and a mix of brick and painted facades. The red-brick pavement is clean and warm-toned, reflecting soft daylight from a blue sky with scattered clouds. Trees rise behind the buildings and cast gentle shade across parts of the road. Planters, flower pots, and climbing greenery bring lush texture to the lower edges of the frame, while bicycles and street furniture add realism without crowding the composition. The overall environment is balanced, sunlit, and serene, resembling a carefully preserved European neighborhood with strong visual structure, detailed materials, and a welcoming public atmosphere.",
    "A calm city street in an old European district is shown under bright, pleasant daylight. Historic townhouses with dark facades, tall windows, and decorative rooflines border a broad brick-paved area that opens gently toward the background. Plants and flowers in pots create a vivid foreground element, especially near the right edge of the frame, while tall leafy trees soften the skyline. A classic lamp post and a few bicycles add everyday urban detail without dominating the composition. Sunlight enters at an angle, producing soft highlights and realistic shadow transitions. The overall feeling is peaceful, spacious, and photorealistic, with a strong sense of place and subtle architectural elegance.",
    "A charming open street scene features red-brick pavement, dark historic buildings, and a combination of carefully arranged greenery and natural tree cover. The architecture is vertical and narrow, with tall windows, pitched roofs, and refined details that suggest an old but well-kept neighborhood. In the foreground, flowers and shrubs in large pots add layered texture and color. Toward the center, a traditional lamp and a few bicycles establish a calm urban daily-life atmosphere. Sunlight warms the facades and pavement, while the sky remains crisp and blue with soft clouds. The composition feels balanced, inviting, and realistic, like a cinematic establishing shot of a peaceful European street.",
    "A serene afternoon street in a quaint European quarter is lined with elegant houses, trimmed greenery, and richly textured pavement. The buildings vary slightly in height and facade treatment, giving the street character and visual rhythm. A patch of bright sunlight falls across the left side of the scene, producing natural contrast against shaded areas under the trees. Decorative flower arrangements occupy the lower right foreground and help frame the composition. Bicycles are visible near the buildings, and a classic lamp post stands upright near the center of the space. The environment is open, warm, and lived-in, with a peaceful atmosphere suited to realistic street-view generation.",
    "A historic neighborhood street appears under clear daylight, with narrow homes, leafy trees, and brick paving arranged in a calm and orderly composition. The buildings feature traditional windows, sloping roofs, and dark facades that contrast beautifully with the vivid green plants placed along the street. A line of sunlight stretches across the open square area, making the red paving stones appear warm and textured. Near the foreground, potted flowers and shrubs add saturated detail and depth. Parked bicycles and a streetlamp help define the space as urban but relaxed. The overall mood is quiet, refined, and visually rich, combining architecture and greenery in a balanced, cinematic way.",
    "A bright and welcoming old-town street is shown from a pedestrian perspective, with brick pavement leading through a tranquil urban square bordered by classic houses and trees. The facades are tall and narrow, painted in darker tones with light catching on window frames and roof edges. On one side, a collection of flowers and leafy plants in pots creates a vivid natural foreground. The opposite side is defined by building walls, parked bicycles, and subtle door and window details. A traditional lamp post stands as a vertical accent near the center. The street feels clean, inviting, and peaceful, with soft afternoon light and realistic architectural detail throughout.",
    "A peaceful residential street in a historic European setting features warm red paving, elegant narrow homes, and mature trees glowing in sunlight. The scene opens into a modest square that feels spacious but intimate, enclosed by architecture and softened by greenery. Windows, rooflines, and decorative trims contribute subtle geometric detail to the facades. Near the front of the image, flowerpots and plants provide layered textures and bright natural color. Bicycles resting near building edges suggest everyday activity without disturbing the calm atmosphere. A blue sky with light clouds completes the setting, giving the entire scene a bright, airy, and photorealistic quality.",
    "A charming cobbled street extends through a quiet town center lined with well-preserved houses, abundant greenery, and decorative urban details. The road surface is composed of red-toned paving stones that catch the sunlight and guide the eye through the frame. Buildings on both sides have tall windows and traditional rooflines, creating a harmonious street wall. Flower arrangements and potted plants enrich the foreground, especially near one side of the image, while a classic lamp post and a few bicycles reinforce a lived-in urban character. The light is soft and warm, the sky is clear, and the whole setting feels calm, intimate, and visually coherent.",
    "A wide pedestrian-friendly European street is illuminated by afternoon sun, revealing richly textured paving stones, elegant facades, and abundant greenery. The houses along the street are narrow and vertical, with old-world roofs and dark-painted surfaces that absorb and reflect light in subtle ways. On the right side, clusters of flowers and potted plants add vivid detail and soften the hard geometry of the buildings. Trees rise behind the rooftops and cast patterned shadows across the open space. A traditional lamp post and bicycles placed near the architecture emphasize realism and scale. The scene feels warm, timeless, and inviting, with a calm urban atmosphere and balanced composition.",
    "A quiet old-town square connected to a narrow street is shown in bright daylight, bordered by historic townhouses and framed by mature trees. The ground is paved with warm red bricks that lead through the composition with strong visual structure. Decorative vegetation in pots adds lush texture at the foreground edge, while bicycles and a classic lamppost contribute recognizable urban details. The buildings feature tall windows, steep roofs, and dark facades that create contrast with the bright sky and sunlit greenery. The scene feels peaceful, clean, and cinematic, combining historical architecture and natural elements in a realistic way suitable for street-scene generation.",
    "A sunlit urban lane in a historic district is lined with carefully maintained brick buildings, green plants, and soft public-square elements. The architecture has a timeless character, with narrow facades, vertical windows, and rooflines stepping gently across the skyline. The foreground contains abundant flowers and leaves, creating a richly detailed border to the scene. Further back, bicycles and a traditional streetlamp establish the calm rhythm of daily city life. Trees provide shade and visual softness, while sunlight warms the pavement and upper floors of the buildings. The environment feels open, inviting, and highly realistic, like a peaceful European street captured in perfect afternoon light.",
    "A picturesque street view reveals old brick paving, elegant narrow homes, and lush greenery arranged within a calm city atmosphere. The buildings vary subtly in tone and shape, with tall facades and historic details that create depth and rhythm. On the side of the frame, vibrant flowers and leafy planters fill the foreground, while a lamppost and bicycles add recognizable urban objects to the middle ground. Mature trees and a clear sky frame the upper portion of the composition, giving the scene a bright, airy feeling. Sunlight filters across the open square and creates natural highlights, making the whole street appear warm, charming, and quietly alive.",
    "A calm neighborhood street in a historic European town is presented with strong architectural detail, warm daylight, and carefully balanced greenery. Brick paving covers the open ground plane, while the surrounding houses display tall windows, old rooflines, and dark facades accented by subtle reflections. Potted flowers and shrubs enrich the front of the scene with natural color and depth. A vintage-style streetlamp stands near the central area, and bicycles near the walls suggest ordinary urban life. Trees rise above the rooftops and contribute soft shade and texture to the background. The entire environment is quiet, welcoming, and realistic, with a peaceful street-level perspective.",
    "A charming city street opens beneath a bright sky, revealing a composition of red-brick paving, historic buildings, and fresh green foliage. The houses are tall and narrow with classic proportions, arranged around a small square that feels intimate and well-preserved. Decorative plants and flowers occupy one side of the foreground, adding layered detail and natural softness. In the middle distance, a lamp post and bicycles lend scale and realism to the urban setting. Sunlight reaches into the frame from one side, creating a gentle glow on the pavement and facades. The overall scene feels warm, elegant, and calm, combining urban structure with inviting natural elements.",
    "A tranquil old-town lane is shown under soft afternoon sun, with traditional townhouses, trees, and flowerpots composing a richly detailed urban environment. The red paving stones guide the eye through the frame and reinforce the calm geometry of the street. Buildings display tall windows, dark facades, and historic roof silhouettes, while bicycles resting against the walls contribute to a leisurely everyday feel. A classic lamp post adds a vertical anchor near the center. Abundant greenery appears both in pots and in mature trees behind the buildings, making the space feel lively but not crowded. The image has a peaceful, realistic, and cinematic character.",
    "A beautiful European street scene features historic architecture, warm sunlight, and carefully placed greenery around a brick-paved square. The surrounding houses have slender proportions, steep roofs, and tall windows, creating a clear visual rhythm along the street. The right foreground is filled with potted flowers and shrubs that add texture and vibrant color. A traditional streetlamp and a few bicycles give the setting a recognizable public-street identity. Trees in the background filter the light and soften the edges of the buildings. The overall mood is calm, inviting, and highly photorealistic, capturing a timeless urban atmosphere with balanced composition and gentle daylight.",
    "A cozy and sunlit residential street is framed by elegant old houses, abundant foliage, and a broad brick surface that opens into a small urban square. The facades are dark and textured, with tall windows catching traces of warm daylight. Potted plants and flowers create a lush foreground corner, while bicycles near the buildings suggest daily life in a quiet neighborhood. A classic lamp post stands prominently within the composition, helping define scale and depth. Trees form a green canopy in the distance, and the sky remains blue with scattered light clouds. The scene feels calm, detailed, and cinematic, with a harmonious mix of architecture and nature.",
    "A peaceful historic street unfolds with warm pavement tones, old-town architecture, and rich greenery illuminated by soft daylight. Narrow houses with steep roofs and tall windows form a textured backdrop to an open brick-paved square. Flowerpots and leafy shrubs frame the lower portion of the scene, especially along one side, while bicycles and a traditional lamp add realistic urban detail. The background trees glow slightly in the afternoon sun, giving the composition both depth and softness. The sky is bright and lightly clouded. Overall, the scene feels inviting, timeless, and realistic, with a balanced interplay of geometry, texture, and natural elements.",
    "A bright European street corner combines historic facades, textured red paving stones, and abundant planted greenery into a calm and appealing urban scene. The buildings have narrow proportions and strong vertical lines, while their roof shapes and window designs suggest a preserved architectural heritage. The foreground contains large flowerpots and dense leaves that add natural color and dimensionality. A classic lamp post rises near the center, and bicycles nearby reinforce the human scale of the environment. Trees border the upper background and filter the sunlight gently. The atmosphere is warm, peaceful, and photorealistic, ideal for representing a quiet, sunlit old-town street.",
    "A refined old-town street scene shows elegant houses, mature trees, and richly detailed paving in warm afternoon light. The facades are dark-toned and vertically proportioned, with window frames and rooflines catching subtle highlights. The street opens slightly into a square-like space, making the composition feel open yet enclosed by architecture. Potted flowers and green plants add visual richness to the foreground, while bicycles and a traditional lamp post provide clear urban context. Sunlight enters from the side and creates gentle glow and shadow variation across the pavement. The overall impression is calm, cinematic, and inviting, with strong realism and a timeless European atmosphere.",
    "A serene brick-paved street lined with historic houses and lush greenery stretches into a softly sunlit urban square. The architecture features narrow facades, tall windows, and roof forms typical of an old European neighborhood. On the right side of the frame, flowers and potted plants create a vivid foreground cluster, while a lamp post and bicycles establish an everyday city-street identity. Trees rise beyond the buildings and soften the skyline with organic shapes and filtered light. The pavement is warm and textured, the sky is bright and clear, and the overall environment feels charming, peaceful, and highly realistic from a street-level point of view.",
    "A picturesque daytime street in a historic district is framed by old houses, mature trees, and decorative greenery arranged across a red-brick public space. The buildings are tall and narrow, with classic windows and dark facades that contrast beautifully against the bright sky. A flower-filled foreground corner adds color and close-up detail, while bicycles and a vintage-style streetlamp contribute realism and scale. Soft sunlight grazes the scene from one side, bringing warmth to the pavement and subtle highlights to the architecture. The entire image feels quiet, elegant, and welcoming, combining preserved urban character with the freshness of natural foliage.",
    "A calm and inviting European street is shown with warm paving stones, dark historic facades, and layers of green plants illuminated by bright afternoon light. The composition opens toward a small square framed by elegant townhouses and tall trees. Decorative planters bring flowers and foliage into the foreground, while bicycles and a classic streetlamp mark the space as a real, lived-in neighborhood. Light falls gently across the buildings and pavement, creating realistic shadow transitions and strong material texture. The sky is blue with a few soft clouds, and the mood is peaceful, photorealistic, and ideal for a cinematic urban street scene.",
    "A timeless old-town street scene features a spacious red-brick roadway, elegant narrow homes, and abundant foliage under a bright blue sky. Tall windows, sloped roofs, and dark painted facades give the architecture visual identity and depth. The foreground is enriched by colorful flowers and potted plants that add softness and close-range texture. In the middle ground, bicycles and a traditional lamp post provide recognizable urban elements. Trees behind the buildings frame the scene with lush green canopies. Sunlight enters at an angle and casts warm highlights across the pavement and facades, producing a tranquil, welcoming atmosphere with strong realism and compositional balance."
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

    # if random.random() < 0.4:
    #     return 1
    return random.choice([1, 9, 17, 25, 33, 41])


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


def make_camera_embedding_static(camera_embedding: torch.Tensor) -> torch.Tensor:
    """
    camera_embedding: [T, 13]
      - first 12 dims: flattened 3x4 relative pose
      - last 1 dim: mask

    Replace all camera motions with identity relative pose (static camera),
    while preserving the original mask.
    """
    assert camera_embedding.dim() == 2 and camera_embedding.shape[1] == 13, \
        f"Expected [T,13], got {camera_embedding.shape}"

    out = camera_embedding.clone()

    # identity relative pose: pose[:3, :] from np.eye(4)
    # [[1,0,0,0],
    #  [0,1,0,0],
    #  [0,0,1,0]]
    static_pose = torch.tensor(
        [1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0],
        device=out.device,
        dtype=out.dtype,
    )

    out[:, :12] = static_pose.unsqueeze(0).expand(out.shape[0], -1)
    return out

def run_attack_forward_once(
    pipe: WanVideoAstraPipeline,
    model_dtype: torch.dtype,
    adv_first_frame: torch.Tensor,
    original_first_lat: torch.Tensor,
    prompt_pool: List[Dict[str, torch.Tensor]],
    num_scheduler_steps: int = 20,
    target_frames: int = 8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    One randomized training step:
      - sample history length
      - sample prompt
      - sample cam_type
      - sample scheduler timestep
      - sample target noise
      - build losses
    """
    device = adv_first_frame.device
    history_length = sample_history_length()
    # cam_type = random.randint(1, 7)
    cam_type = 1
    prompt_emb = sample_prompt_emb(prompt_pool)

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
    camera_embedding_full = generate_sekai_camera_embeddings_sliding(
        cam_data=None,
        start_frame=0,
        initial_condition_frames=history_length,
        new_frames=target_frames,
        total_generated=0,
        use_real_poses=False,
        cam_type=cam_type,   # 这里保留即可，后面会覆盖掉
    ).to(device=device, dtype=model_dtype)

    # Force static camera trajectory in attack
    camera_embedding_full = make_camera_embedding_static(camera_embedding_full)

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
    t_idx = random.randint(0,50)
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

    # ============================================================
    # [MODIFIED] Loss:
    # ground-truth velocity = x1(context image latent) - x0(noisy target latent)
    #
    # x1: clean first-frame latent, repeated over target frames
    # x0: current noisy target latents = new_latents
    #
    # We minimize ||pred_v - gt_v||^2
    # ============================================================
    target_x1 = original_first_lat.to(device=new_latents.device, dtype=new_latents.dtype).unsqueeze(0).repeat(1, 1, target_frames, 1, 1)   # [1,16,T,60,104]
    # gt_velocity = target_x1 - new_latents                                        # [1,16,T,60,104]
    gt_velocity = new_latents-target_x1 

    loss_vel = (pred_v.float() - gt_velocity.float()).pow(2).mean()

    pred_v_norm = pred_v.float().pow(2).mean()
    gt_v_norm = gt_velocity.float().pow(2).mean()
    # print("pred_v_norm: ", pred_v_norm)
    # print("gt v norm: ", gt_v_norm)

    aux_info = {
        "history_length": history_length,
        "cam_type": cam_type,
        "timestep_index": t_idx,
        "timestep_value": float(timestep.detach().float().cpu().item()),
        "pred_v_shape": tuple(pred_v.shape),
        "gt_v_shape": tuple(gt_velocity.shape),
    }

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

    # best_objective = -1e18
    # best_adv = adv_first_frame.clone().detach()

    for step in range(1, args.num_steps + 1):
        adv_first_frame = adv_first_frame.detach().clone().requires_grad_(True)

        loss_vel, aux_info = run_attack_forward_once(
            pipe=pipe,
            model_dtype=model_dtype,
            adv_first_frame=adv_first_frame,
            original_first_lat=original_first_lat,
            prompt_pool=prompt_pool,
            num_scheduler_steps=args.num_scheduler_steps,
            target_frames=args.target_frames,
        )

        objective = loss_vel
        objective.backward()

        if adv_first_frame.grad is None:
            raise RuntimeError("adv_first_frame.grad is None. Check gradient flow.")

        with torch.no_grad():
            adv_first_frame = adv_first_frame + args.alpha * adv_first_frame.grad.sign()
            adv_first_frame = project_and_clamp(
                adv_x=adv_first_frame,
                original_x=original_first_frame,
                eps=args.eps,
            )

        obj_val = objective.detach().float().item()
        vel_val = loss_vel.detach().float().item()
        # attn_val = loss_attn.detach().float().item()

        # if obj_val > best_objective:
        #     best_objective = obj_val
        #     best_adv = adv_first_frame.clone().detach()

        if step % args.log_every == 0 or step == 1:
            print(
                f"[Step {step:04d}] "
                f"vel={vel_val:.6f} | "
                f"T_hist={aux_info['history_length']} | "
                f"t_idx={aux_info['timestep_index']} | "
            )

        if step % args.save_every == 0 or step == args.num_steps:
            cur_path = os.path.join(args.output_dir, f"adv_step_{step:04d}.png")
            # best_path = os.path.join(args.output_dir, "adv_best.png")
            save_normalized_tensor_as_image(adv_first_frame, cur_path)
            # save_normalized_tensor_as_image(best_adv, best_path)

    final_path = os.path.join(args.output_dir, "adv_final.png")
    save_normalized_tensor_as_image(adv_first_frame, final_path)
    print(f"[Done] saved final adversarial first frame to: {final_path}")


# ============================================================
# 8. CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser("PGD attack on the input first frame for autoregressive Wan-Astra-FramePack-MoE video generation")

    parser.add_argument("--input_image", type=str, default="./examples/condition_images/garden_3.png")
    parser.add_argument("--dit_path", type=str, default="./models/Astra/checkpoints/diffusion_pytorch_model.ckpt")
    parser.add_argument("--wan_model_path", type=str, default="./models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--output_dir", type=str, default="atk_minV2/garden3")

    parser.add_argument("--device", type=str, default="cuda")

    # attack hyperparams
    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument("--eps", type=float, default=0.05,
                        help="L_inf budget in normalized [-1,1] space.")
    parser.add_argument("--alpha", type=float, default=0.004,
                        help="PGD step size in normalized space.")
    # parser.add_argument("--lambda_vel", type=float, default=1.0)
    # parser.add_argument("--lambda_attn", type=float, default=3.0)

    # generation / training randomization
    parser.add_argument("--target_frames", type=int, default=8)
    parser.add_argument("--num_scheduler_steps", type=int, default=1000)
    parser.add_argument("--num_prompts", type=int, default=30)

    # moe config
    parser.add_argument("--moe_num_experts", type=int, default=3)
    parser.add_argument("--moe_top_k", type=int, default=1)
    parser.add_argument("--moe_hidden_dim", type=int, default=None)

    # attention loss compute cost control
    parser.add_argument("--attn_query_chunk", type=int, default=16,
                        help="Chunk size over target queries when computing target->start attention mass. Smaller = safer memory, slower.")

    # logging / saving
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    attack_first_frame_pgd(args)


if __name__ == "__main__":
    main()