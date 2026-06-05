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
    # "A sunlit European street lined with historic buildings and vibrant greenery creates a warm, charming, and inviting atmosphere. The scene shows a picturesque open square paved with red bricks, surrounded by classic narrow townhouses featuring tall windows, gabled roofs, and dark-painted facades. On the right side, a lush arrangement of potted plants and blooming flowers adds rich color and texture to the foreground. A vintage-style streetlamp stands prominently near the center-right, contributing to the timeless character of the street. Mature trees frame the background, their leaves glowing in the warm afternoon sunlight. Bicycles rest along the edges of the buildings, reinforcing the urban yet leisurely feel. The sky is bright blue with scattered clouds, and soft sun flares enter from the left.",
    # "A peaceful old-town street stretches forward beneath a soft afternoon sky, surrounded by elegant brick buildings, leafy trees, and decorative storefronts. Warm sunlight washes across the cobblestone pavement, revealing subtle texture and long shadows. Flower boxes spill from windows, while bicycles lean quietly against stone walls and iron railings. A small cafe terrace appears on one side with neatly arranged chairs and potted plants, suggesting a calm and inviting public space. The facades of the surrounding buildings vary in height and color, giving the street a layered visual rhythm. The entire setting feels quiet, cinematic, and richly detailed, with a balance of architecture, greenery, and open urban space.",
    # "A bright residential street in a charming European neighborhood is framed by narrow townhouses, mature trees, and a clean red-brick roadway. The architecture features tall windows, steep roofs, and decorative trim that create a refined historic atmosphere. Soft sunlight reflects off the upper floors and filters through the leaves, casting dappled shadows on the pavement below. Along the sidewalk, potted flowers and shrubs add bursts of color and texture. A vintage lamppost stands beside a bench near the edge of a small square, giving the scene a timeless public-street character. The air feels fresh, the sky is lightly clouded, and the overall mood is calm, open, and welcoming.",
    # "A quiet urban street opens into a small square bordered by elegant buildings with dark facades, tall windows, and classic rooflines. The brick pavement glows warmly under late afternoon sunlight, while trees and climbing plants soften the architectural edges. A cluster of flowers in decorative pots occupies the foreground, bringing vivid natural detail close to the viewer. Several bicycles are parked casually along the buildings, reinforcing the everyday character of the space. In the distance, more houses and greenery create depth and a sense of continuity. The light is gentle and golden, the sky is bright, and the whole scene feels like a tranquil, photorealistic European street environment.",
    # "A picturesque pedestrian-friendly street is illuminated by warm sunlight and framed by old brick houses, green foliage, and ornamental street elements. The road surface is composed of rich reddish paving stones, leading the eye toward a softly lit urban square. On one side, flowerpots overflow with blossoms and leafy plants, while on the other side, storefront windows and doors create a rhythm of vertical details. Mature trees rise behind the buildings, partially filtering the sun and adding layered shadows. A classic streetlamp and a few parked bicycles contribute to the lived-in atmosphere. The scene feels peaceful, detailed, and ideal for cinematic city-view generation with strong architectural identity.",
    # "A narrow European street bathed in gentle sunshine features historic buildings, colorful plants, and carefully textured paving stones. The composition emphasizes depth, with the street receding toward a bright open square framed by trees and rooftops. The buildings have tall proportions, dark-painted facades, and old-world roof shapes that evoke a timeless urban neighborhood. On the right side, decorative potted plants and flowers fill the foreground with life and color. The left side receives a soft burst of sunlight that creates natural glow and subtle flare. A few bicycles and a traditional lamp anchor the everyday realism of the space, producing a calm, inviting, and richly atmospheric street scene.",
    # "A beautiful town street unfolds between classic narrow houses with tall windows, sloped roofs, and a mix of brick and painted facades. The red-brick pavement is clean and warm-toned, reflecting soft daylight from a blue sky with scattered clouds. Trees rise behind the buildings and cast gentle shade across parts of the road. Planters, flower pots, and climbing greenery bring lush texture to the lower edges of the frame, while bicycles and street furniture add realism without crowding the composition. The overall environment is balanced, sunlit, and serene, resembling a carefully preserved European neighborhood with strong visual structure, detailed materials, and a welcoming public atmosphere.",
    # "A calm city street in an old European district is shown under bright, pleasant daylight. Historic townhouses with dark facades, tall windows, and decorative rooflines border a broad brick-paved area that opens gently toward the background. Plants and flowers in pots create a vivid foreground element, especially near the right edge of the frame, while tall leafy trees soften the skyline. A classic lamp post and a few bicycles add everyday urban detail without dominating the composition. Sunlight enters at an angle, producing soft highlights and realistic shadow transitions. The overall feeling is peaceful, spacious, and photorealistic, with a strong sense of place and subtle architectural elegance.",
    # "A charming open street scene features red-brick pavement, dark historic buildings, and a combination of carefully arranged greenery and natural tree cover. The architecture is vertical and narrow, with tall windows, pitched roofs, and refined details that suggest an old but well-kept neighborhood. In the foreground, flowers and shrubs in large pots add layered texture and color. Toward the center, a traditional lamp and a few bicycles establish a calm urban daily-life atmosphere. Sunlight warms the facades and pavement, while the sky remains crisp and blue with soft clouds. The composition feels balanced, inviting, and realistic, like a cinematic establishing shot of a peaceful European street.",
    # "A serene afternoon street in a quaint European quarter is lined with elegant houses, trimmed greenery, and richly textured pavement. The buildings vary slightly in height and facade treatment, giving the street character and visual rhythm. A patch of bright sunlight falls across the left side of the scene, producing natural contrast against shaded areas under the trees. Decorative flower arrangements occupy the lower right foreground and help frame the composition. Bicycles are visible near the buildings, and a classic lamp post stands upright near the center of the space. The environment is open, warm, and lived-in, with a peaceful atmosphere suited to realistic street-view generation.",
    # "A historic neighborhood street appears under clear daylight, with narrow homes, leafy trees, and brick paving arranged in a calm and orderly composition. The buildings feature traditional windows, sloping roofs, and dark facades that contrast beautifully with the vivid green plants placed along the street. A line of sunlight stretches across the open square area, making the red paving stones appear warm and textured. Near the foreground, potted flowers and shrubs add saturated detail and depth. Parked bicycles and a streetlamp help define the space as urban but relaxed. The overall mood is quiet, refined, and visually rich, combining architecture and greenery in a balanced, cinematic way.",
    # "A bright and welcoming old-town street is shown from a pedestrian perspective, with brick pavement leading through a tranquil urban square bordered by classic houses and trees. The facades are tall and narrow, painted in darker tones with light catching on window frames and roof edges. On one side, a collection of flowers and leafy plants in pots creates a vivid natural foreground. The opposite side is defined by building walls, parked bicycles, and subtle door and window details. A traditional lamp post stands as a vertical accent near the center. The street feels clean, inviting, and peaceful, with soft afternoon light and realistic architectural detail throughout.",
    # "A peaceful residential street in a historic European setting features warm red paving, elegant narrow homes, and mature trees glowing in sunlight. The scene opens into a modest square that feels spacious but intimate, enclosed by architecture and softened by greenery. Windows, rooflines, and decorative trims contribute subtle geometric detail to the facades. Near the front of the image, flowerpots and plants provide layered textures and bright natural color. Bicycles resting near building edges suggest everyday activity without disturbing the calm atmosphere. A blue sky with light clouds completes the setting, giving the entire scene a bright, airy, and photorealistic quality.",
    # "A charming cobbled street extends through a quiet town center lined with well-preserved houses, abundant greenery, and decorative urban details. The road surface is composed of red-toned paving stones that catch the sunlight and guide the eye through the frame. Buildings on both sides have tall windows and traditional rooflines, creating a harmonious street wall. Flower arrangements and potted plants enrich the foreground, especially near one side of the image, while a classic lamp post and a few bicycles reinforce a lived-in urban character. The light is soft and warm, the sky is clear, and the whole setting feels calm, intimate, and visually coherent.",
    # "A wide pedestrian-friendly European street is illuminated by afternoon sun, revealing richly textured paving stones, elegant facades, and abundant greenery. The houses along the street are narrow and vertical, with old-world roofs and dark-painted surfaces that absorb and reflect light in subtle ways. On the right side, clusters of flowers and potted plants add vivid detail and soften the hard geometry of the buildings. Trees rise behind the rooftops and cast patterned shadows across the open space. A traditional lamp post and bicycles placed near the architecture emphasize realism and scale. The scene feels warm, timeless, and inviting, with a calm urban atmosphere and balanced composition.",
    # "A quiet old-town square connected to a narrow street is shown in bright daylight, bordered by historic townhouses and framed by mature trees. The ground is paved with warm red bricks that lead through the composition with strong visual structure. Decorative vegetation in pots adds lush texture at the foreground edge, while bicycles and a classic lamppost contribute recognizable urban details. The buildings feature tall windows, steep roofs, and dark facades that create contrast with the bright sky and sunlit greenery. The scene feels peaceful, clean, and cinematic, combining historical architecture and natural elements in a realistic way suitable for street-scene generation.",
    # "A sunlit urban lane in a historic district is lined with carefully maintained brick buildings, green plants, and soft public-square elements. The architecture has a timeless character, with narrow facades, vertical windows, and rooflines stepping gently across the skyline. The foreground contains abundant flowers and leaves, creating a richly detailed border to the scene. Further back, bicycles and a traditional streetlamp establish the calm rhythm of daily city life. Trees provide shade and visual softness, while sunlight warms the pavement and upper floors of the buildings. The environment feels open, inviting, and highly realistic, like a peaceful European street captured in perfect afternoon light.",
    # "A picturesque street view reveals old brick paving, elegant narrow homes, and lush greenery arranged within a calm city atmosphere. The buildings vary subtly in tone and shape, with tall facades and historic details that create depth and rhythm. On the side of the frame, vibrant flowers and leafy planters fill the foreground, while a lamppost and bicycles add recognizable urban objects to the middle ground. Mature trees and a clear sky frame the upper portion of the composition, giving the scene a bright, airy feeling. Sunlight filters across the open square and creates natural highlights, making the whole street appear warm, charming, and quietly alive.",
    # "A calm neighborhood street in a historic European town is presented with strong architectural detail, warm daylight, and carefully balanced greenery. Brick paving covers the open ground plane, while the surrounding houses display tall windows, old rooflines, and dark facades accented by subtle reflections. Potted flowers and shrubs enrich the front of the scene with natural color and depth. A vintage-style streetlamp stands near the central area, and bicycles near the walls suggest ordinary urban life. Trees rise above the rooftops and contribute soft shade and texture to the background. The entire environment is quiet, welcoming, and realistic, with a peaceful street-level perspective.",
    # "A charming city street opens beneath a bright sky, revealing a composition of red-brick paving, historic buildings, and fresh green foliage. The houses are tall and narrow with classic proportions, arranged around a small square that feels intimate and well-preserved. Decorative plants and flowers occupy one side of the foreground, adding layered detail and natural softness. In the middle distance, a lamp post and bicycles lend scale and realism to the urban setting. Sunlight reaches into the frame from one side, creating a gentle glow on the pavement and facades. The overall scene feels warm, elegant, and calm, combining urban structure with inviting natural elements.",
    # "A tranquil old-town lane is shown under soft afternoon sun, with traditional townhouses, trees, and flowerpots composing a richly detailed urban environment. The red paving stones guide the eye through the frame and reinforce the calm geometry of the street. Buildings display tall windows, dark facades, and historic roof silhouettes, while bicycles resting against the walls contribute to a leisurely everyday feel. A classic lamp post adds a vertical anchor near the center. Abundant greenery appears both in pots and in mature trees behind the buildings, making the space feel lively but not crowded. The image has a peaceful, realistic, and cinematic character.",
    # "A beautiful European street scene features historic architecture, warm sunlight, and carefully placed greenery around a brick-paved square. The surrounding houses have slender proportions, steep roofs, and tall windows, creating a clear visual rhythm along the street. The right foreground is filled with potted flowers and shrubs that add texture and vibrant color. A traditional streetlamp and a few bicycles give the setting a recognizable public-street identity. Trees in the background filter the light and soften the edges of the buildings. The overall mood is calm, inviting, and highly photorealistic, capturing a timeless urban atmosphere with balanced composition and gentle daylight.",
    # "A cozy and sunlit residential street is framed by elegant old houses, abundant foliage, and a broad brick surface that opens into a small urban square. The facades are dark and textured, with tall windows catching traces of warm daylight. Potted plants and flowers create a lush foreground corner, while bicycles near the buildings suggest daily life in a quiet neighborhood. A classic lamp post stands prominently within the composition, helping define scale and depth. Trees form a green canopy in the distance, and the sky remains blue with scattered light clouds. The scene feels calm, detailed, and cinematic, with a harmonious mix of architecture and nature.",
    # "A peaceful historic street unfolds with warm pavement tones, old-town architecture, and rich greenery illuminated by soft daylight. Narrow houses with steep roofs and tall windows form a textured backdrop to an open brick-paved square. Flowerpots and leafy shrubs frame the lower portion of the scene, especially along one side, while bicycles and a traditional lamp add realistic urban detail. The background trees glow slightly in the afternoon sun, giving the composition both depth and softness. The sky is bright and lightly clouded. Overall, the scene feels inviting, timeless, and realistic, with a balanced interplay of geometry, texture, and natural elements.",
    # "A bright European street corner combines historic facades, textured red paving stones, and abundant planted greenery into a calm and appealing urban scene. The buildings have narrow proportions and strong vertical lines, while their roof shapes and window designs suggest a preserved architectural heritage. The foreground contains large flowerpots and dense leaves that add natural color and dimensionality. A classic lamp post rises near the center, and bicycles nearby reinforce the human scale of the environment. Trees border the upper background and filter the sunlight gently. The atmosphere is warm, peaceful, and photorealistic, ideal for representing a quiet, sunlit old-town street.",
    # "A refined old-town street scene shows elegant houses, mature trees, and richly detailed paving in warm afternoon light. The facades are dark-toned and vertically proportioned, with window frames and rooflines catching subtle highlights. The street opens slightly into a square-like space, making the composition feel open yet enclosed by architecture. Potted flowers and green plants add visual richness to the foreground, while bicycles and a traditional lamp post provide clear urban context. Sunlight enters from the side and creates gentle glow and shadow variation across the pavement. The overall impression is calm, cinematic, and inviting, with strong realism and a timeless European atmosphere.",
    # "A serene brick-paved street lined with historic houses and lush greenery stretches into a softly sunlit urban square. The architecture features narrow facades, tall windows, and roof forms typical of an old European neighborhood. On the right side of the frame, flowers and potted plants create a vivid foreground cluster, while a lamp post and bicycles establish an everyday city-street identity. Trees rise beyond the buildings and soften the skyline with organic shapes and filtered light. The pavement is warm and textured, the sky is bright and clear, and the overall environment feels charming, peaceful, and highly realistic from a street-level point of view.",
    # "A picturesque daytime street in a historic district is framed by old houses, mature trees, and decorative greenery arranged across a red-brick public space. The buildings are tall and narrow, with classic windows and dark facades that contrast beautifully against the bright sky. A flower-filled foreground corner adds color and close-up detail, while bicycles and a vintage-style streetlamp contribute realism and scale. Soft sunlight grazes the scene from one side, bringing warmth to the pavement and subtle highlights to the architecture. The entire image feels quiet, elegant, and welcoming, combining preserved urban character with the freshness of natural foliage.",
    # "A calm and inviting European street is shown with warm paving stones, dark historic facades, and layers of green plants illuminated by bright afternoon light. The composition opens toward a small square framed by elegant townhouses and tall trees. Decorative planters bring flowers and foliage into the foreground, while bicycles and a classic streetlamp mark the space as a real, lived-in neighborhood. Light falls gently across the buildings and pavement, creating realistic shadow transitions and strong material texture. The sky is blue with a few soft clouds, and the mood is peaceful, photorealistic, and ideal for a cinematic urban street scene.",
    # "A timeless old-town street scene features a spacious red-brick roadway, elegant narrow homes, and abundant foliage under a bright blue sky. Tall windows, sloped roofs, and dark painted facades give the architecture visual identity and depth. The foreground is enriched by colorful flowers and potted plants that add softness and close-range texture. In the middle ground, bicycles and a traditional lamp post provide recognizable urban elements. Trees behind the buildings frame the scene with lush green canopies. Sunlight enters at an angle and casts warm highlights across the pavement and facades, producing a tranquil, welcoming atmosphere with strong realism and compositional balance."
    "A winding mountain road cuts through a serene alpine landscape, framed by snow-capped peaks, dense forests, and distant structures under a soft, overcast sky. The scene depicts a road winding through a mountainous landscape. Snow-capped peaks rise in the background under a cloudy sky. Lush green trees and vegetation line the roadside. On the right, there are buildings and parked trucks. The overall atmosphere is serene, with a sense of remoteness and natural beauty. The lighting is soft due to the overcast sky, casting a muted tone over the entire scene.",
    "A twisting road meanders through a peaceful alpine scenery, bordered by snowy summits, thick woodlands, and far-off buildings beneath a gentle, cloudy sky.",
    "This picturesque mountain route navigates a tranquil landscape, surrounded by towering snow-laden peaks, rich forests, and distant structures under a softly filtered sky.",
    "A serene alpine road winds its way through a breathtaking landscape, with snow-covered mountains rising high above, dense greenery lining the path, and structures visible in the background.",
    "In this calm mountain scene, a curving road traverses the lush landscape, with magnificent snow-capped peaks soaring above and buildings nestled among the trees.",
    "An idyllic mountain highway snakes through a peaceful setting, framed by white-capped mountains, verdant forests, and structures nestled in the distance under a cloudy sky.",
    "The winding mountain road offers a charming view of an alpine landscape, with frosty peaks towering in the background and a serene atmosphere enveloping the greenery and buildings nearby.",
    "This image captures a meandering road through a tranquil mountain scene, featuring snow-covered summits, abundant trees, and distant buildings, all beneath a soothing overcast sky.",
    "A gentle alpine road winds through a serene landscape, flanked by snow-draped peaks and lush green trees, with buildings and vehicles visible in the distance.",
    "The road twists gracefully through the quiet mountains, surrounded by majestic snow-capped peaks, dense forests, and distant structures beneath a soft, cloudy canopy.",
    "A scenic highway weaves through a tranquil alpine environment, characterized by towering snowy mountains, thick vegetation, and distant buildings bathed in gentle light.",
    "This image portrays a winding road cutting through a peaceful mountain backdrop, with snow-clad peaks, lush trees lining the way, and distant structures under a soft sky.",
    "A tranquil road navigates the mountainous terrain, embraced by snow-capped peaks and dense woods, with structures visible in the background under an overcast sky.",
    "A winding roadway curves through a serene landscape of the Alps, with snow-capped mountains rising majestically in the background and green trees flanking the sides.",
    "In this quiet mountain scene, a winding path leads through an enchanting setting, surrounded by frosty peaks, vibrant woods, and distant buildings under a gentle sky.",
    "The road bends gracefully through a serene alpine vista, framed by snow-capped summits, rich forests, and distant structures beneath a soft, overcast sky.",
    "An elegant mountain road meanders through a peaceful landscape, flanked by impressive snowy peaks and dense greenery, with visible structures in the distance.",
    "The winding path through the mountains captures a serene atmosphere, with snowy peaks in the background, vibrant forests along the road, and distant buildings emerging softly.",
    "A picturesque road cuts through a serene alpine landscape, lined with snow-dusted mountains and lush trees, where distant structures add to the tranquil charm.",
    "This image showcases a winding mountain route set against snow-capped peaks and dense forests, with structures peeking through the greenery beneath a soft sky.",
    "A winding mountain path graces a tranquil alpine scene, surrounded by snow-covered heights and flourishing trees, with distant buildings dotting the landscape.",
    "In this serene landscape, a curved road winds through majestic mountains, with snow-capped peaks towering above and structures gently nestled nearby.",
    "The road winds through a picturesque alpine environment, set among majestic, snow-capped peaks and rich forests, with buildings and vehicles visible on the side.",
    "A winding road traverses a peaceful alpine landscape, framed by snowy mountains, lush trees, and distant buildings under a gentle, cloud-covered sky.",
    "This winding road showcases a vibrant mountain landscape, surrounded by snow-capped peaks, rich forests, and buildings visible in the distance under soft skies.",
    "The road elegantly weaves through a serene mountainous landscape, embraced by white-capped peaks and dense greenery, with structures suggesting human presence nearby.",
    "In this tranquil alpine scene, a winding road connects distant structures amidst a backdrop of snow-covered peaks and lush trees beneath a soft overcast sky.",
    "A curving mountain road flows through a peaceful landscape, with snow-crowned peaks in the distance, vibrant forests along the path, and structures subtly present.",
    "A winding highway leads through a serene mountain setting, featuring snow-covered summits, lush greenery beside the road, and distant structures beneath a cloudy sky.",
    "A twisting road cuts through the calmness of a mountainous terrain, where snow-topped peaks meet dense woodlands, and distant structures provide a hint of civilization.",
    "This image presents a serene winding road through a picturesque mountain landscape, surrounded by snowy peaks and thick forests, with buildings nestled away in the distance."

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

    # if random.random() < 0.5:
    #     return 1
    return random.choice([1, 9, 17, 25, 33, 41])


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
            adv_first_frame = adv_first_frame + args.alpha * adv_first_frame.grad.sign()
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

    parser.add_argument("--input_image", type=str, default="./examples/condition_images/garden_3.png")
    parser.add_argument("--dit_path", type=str, default="./models/Astra/checkpoints/diffusion_pytorch_model.ckpt")
    parser.add_argument("--wan_model_path", type=str, default="./models/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--output_dir", type=str, default="atk_minV2/garden3")

    parser.add_argument("--device", type=str, default="cuda")

    # attack hyperparams
    parser.add_argument("--num_steps", type=int, default=400)
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