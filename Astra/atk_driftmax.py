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

from scripts.utils import (
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
    "A dramatic mountain vista under a blue sky.",    
    "A breathtaking view of towering mountains against a clear blue sky, where the peaks are adorned with snow and evergreen trees add richness to the scenery.",
    "The majestic mountain range rises impressively under a vivid blue sky, showcasing snow-capped summits and a backdrop of fluffy clouds.",
    "An awe-inspiring landscape featuring snow-topped mountains, azure skies, and verdant evergreens, all contributing to a stunning natural panorama.",
    "This striking mountain vista, framed by a brilliant blue sky and dotted with soft clouds, highlights the beauty of the snow-covered peaks and lush trees.",
    "Under a clear blue sky, the grand mountains stand tall with their snow-capped tops, while evergreen trees at their base provide a sense of scale.",
    "A captivating mountain scene unfolds beneath a bright blue expanse, where snowy peaks tower majestically and fluffy clouds float by.",
    "The fantastic mountain view showcases a vast range, partially veiled in snow, contrasted beautifully against the deep blue sky and soft white clouds.",
    "This stunning mountain panorama features pristine snow peaks under a sapphire sky, complemented by an array of evergreen trees populating the lower slopes.",
    "A magnificent vista of snow-clad mountains meets a clear blue sky, revealing the enchanting contrast of evergreen trees and fluffy clouds.",
    "The scene is alive with majestic snowy peaks, a brilliant azure sky, and fluffy clouds, creating an inspiring landscape of natural beauty.",
    "Snow-capped mountains rise dramatically beneath a vibrant blue sky, with evergreen forests creating a lush foreground that enhances the vista.",
    "A picturesque view captures the grandeur of a mountain range, crowned with snow against a bright sky, and framed by lush evergreens.",
    "Set under a bright blue canopy, the awe-inspiring mountain vista features soaring peaks dusted with snow and a backdrop of soft, fluffy clouds.",
    "This compelling scene presents a majestic mountain range, partially shrouded in snow, set against a vast blue sky filled with cottony clouds.",
    "A panoramic view of inspiring mountains, their snowy tops gleaming under a clear blue sky, while evergreens line the base, enriching the landscape.",
    "The impressive mountain landscape, accentuated by snow-capped heights and a cloud-strewn sky, evokes a sense of wonder and grandeur.",
    "With their snow-covered peaks reaching for the azure sky, these majestic mountains stand tall beside evergreen trees, framing a breathtaking view.",
    "A dramatic portrayal of nature, featuring snow-topped peaks against a vivid blue sky, where plush clouds and evergreens add depth to the scene.",
    "Beneath a brilliant blue sky, the awe-inspiring mountain vista reveals a majestic range with snow-dusted peaks and lush evergreen trees at its base.",
    "This scenic view highlights dramatic snow-capped mountains rising against a striking blue sky, with fluffy clouds floating gracefully above.",
    "An impressive mountain range, draped in snow and framed by a brilliant blue sky, is complemented by the lush greenery of evergreen trees.",
    "The grandeur of the mountains is emphasized by their snow-covered peaks, standing regal under a clear blue sky dotted with wispy clouds.",
    "A stunning landscape emerges, showcasing snow-capped mountains beneath a radiant blue sky, embellished by fluffy clouds and verdant trees.",
    "A remarkable view of the mountains reveals majestic peaks cloaked in snow, under a vast blue sky filled with clouds that enhance the natural beauty.",
    "This striking mountain landscape features a magnificent range adorned with snow, set against a brilliant blue sky and lush evergreen foliage below.",
    "The snowy summits of these grand mountains tower under an expansive blue sky, creating a breathtaking scene enhanced by evergreen trees.",
    "An awe-inspiring view unfolds with snow-capped peaks against a clear blue sky, while evergreen forests line the foot of the majestic mountains.",
    "Majestic mountains rise gracefully beneath a vibrant blue sky, their snowy heights contrasting with the soft clouds and lush green trees below.",
    "An impressive mountain landscape stretches beneath a clear blue sky, featuring towering, snow-dusted summits, lush evergreen forests, and soft, billowy clouds that enhance the scene's breathtaking beauty.",
    "Beneath a vivid blue sky, a stunning mountain panorama reveals its snow-covered peaks, surrounded by verdant pines and wispy clouds, creating an inspiring natural spectacle."
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
        cam_type=cam_type,   
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
    parser.add_argument("--output_dir", type=str, default="atk_minV")

    parser.add_argument("--device", type=str, default="cuda")

    # attack hyperparams
    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument("--eps", type=float, default=0.05,
                        help="L_inf budget in normalized [-1,1] space.")
    parser.add_argument("--alpha", type=float, default=0.004,
                        help="PGD step size in normalized space.")

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