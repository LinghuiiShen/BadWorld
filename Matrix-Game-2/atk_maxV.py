import os
import sys
import json
import random
import argparse
from typing import Dict, Tuple

import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import v2
from safetensors.torch import load_file
from diffusers.utils import load_image

# -----------------------------
# Project imports (Matrix-Game-2)
# -----------------------------
from utils.misc import set_seed
from utils.conditions import Bench_actions_gta_drive
from utils.wan_wrapper import WanDiffusionWrapper
from wan.vae.wanx_vae import get_wanx_vae_wrapper

# ============================================================
# 0. Logging helper
# ============================================================

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
# 1. Preprocess / image utils
# ============================================================

def build_frame_process():
    # 与 inference.py 对齐
    return v2.Compose([
        v2.Resize(size=(352, 640), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def resizecrop_pil(image: Image.Image, th: int = 352, tw: int = 640) -> Image.Image:
    w, h = image.size
    if h / w > th / tw:
        new_w = int(w)
        new_h = int(new_w * th / tw)
    else:
        new_h = int(h)
        new_w = int(new_h * tw / th)

    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    image = image.crop((left, top, right, bottom))
    return image


def load_condition_image_tensor(image_path: str, device: str) -> torch.Tensor:
    """
    Returns:
        [1, 3, 352, 640], normalized to [-1, 1]
    """
    image = load_image(image_path)
    image = resizecrop_pil(image, 352, 640)
    frame_process = build_frame_process()
    x = frame_process(image).unsqueeze(0).to(device=device, dtype=torch.float32)
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
# 2. Model loading / attack core
# ============================================================

class FirstFramePGDAttack:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16
        self.vae_dtype = torch.float16
        self.frame_process = build_frame_process()

        self._init_config()
        self._init_models()

    def _init_config(self):
        self.config = OmegaConf.load(self.args.config_path)

    def _init_models(self):
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}),
            is_causal=True,
        )

        cfg_num_frame_per_block = int(getattr(self.config, "num_frame_per_block", 1))
        if self.args.override_num_frame_per_block is not None:
            generator.model.num_frame_per_block = int(self.args.override_num_frame_per_block)
        elif cfg_num_frame_per_block > 1:
            generator.model.num_frame_per_block = cfg_num_frame_per_block

        if self.args.checkpoint_path:
            print("Loading pretrained generator checkpoint...")
            state_dict = load_file(self.args.checkpoint_path)
            generator.load_state_dict(state_dict)

        self.generator = generator.to(device=self.device, dtype=self.weight_dtype)
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad_(False)

        # VAE / CLIP wrapper
        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, self.vae_dtype)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

        self.scheduler = self.generator.get_scheduler()

        self.frame_seq_length = 44 * 80 // (2 * 2)  # = 880
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"[Info] model.num_frame_per_block = {self.generator.model.num_frame_per_block}")
        print(f"[Info] local_attn_size = {self.local_attn_size}")
        print(f"[Info] frame_seq_length = {self.frame_seq_length}")

    def reset_model_block_masks(self):
        self.generator.model.block_mask = None
        self.generator.model.block_mask_keyboard = None
        self.generator.model.block_mask_mouse = None

    def encode_single_frame_to_latent(self, frame_4d: torch.Tensor) -> torch.Tensor:
        """
        frame_4d: [1,3,352,640] in [-1,1]
        returns:
            [1,16,1,44,80]
        """
        assert frame_4d.ndim == 4 and frame_4d.shape[0] == 1
        frame_5d = frame_4d[:, :, None, :, :].to(device=self.device, dtype=self.weight_dtype)
        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        lat = self.vae.encode(frame_5d, device=self.device, **tiler_kwargs).to(
            device=self.device, dtype=self.weight_dtype
        )
        return lat


    def build_cond_concat(self, adv_first_frame: torch.Tensor, total_latent_frames: int):
        # adv_first_frame: [1,3,352,640] in [-1,1]
        image_5d = adv_first_frame[:, :, None, :, :].to(device=self.device, dtype=self.weight_dtype)

        padding_video = torch.zeros_like(image_5d).repeat(
            1, 1, 4 * (total_latent_frames - 1), 1, 1
        )
        img_cond_pixels = torch.cat([image_5d, padding_video], dim=2)

        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond_pixels, device=self.device, **tiler_kwargs).to(
            device=self.device, dtype=self.weight_dtype
        )

        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
        return cond_concat

    def build_conditional_dict(
        self,
        adv_first_frame: torch.Tensor,
        window_start_frame: int,
        total_length: int,
        target_start_frame: int,
        target_length: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Build local conditional_dict for a local window [window_start_frame : window_start_frame + total_length).

        Important:
        - cond_concat is first built in global time, then sliced locally
        - action is also aligned to the local window's absolute position
        """

        global_length = window_start_frame + total_length

        cond_concat = self.clean_cond_concat_global[:, :, window_start_frame: window_start_frame + total_length].clone()

        adv_first_lat = self.encode_single_frame_to_latent(adv_first_frame)   # [1,16,1,44,80]

        if window_start_frame == 0:
            cond_concat[:, 4:, 0:1] = adv_first_lat

        adv_first_frame_5d = adv_first_frame[:, :, None, :, :].to(
            self.device, self.weight_dtype
        )
        visual_context = self.vae.clip.encode_video(adv_first_frame_5d)
        num_real_frames_global = 1 + 4 * (global_length - 1)
        cond_data = Bench_actions_gta_drive(num_real_frames_global)

        action_prefix_real_len = 1 + 4 * (target_start_frame + target_length - 1)

        mouse_condition = cond_data["mouse_condition"][:action_prefix_real_len].unsqueeze(0).to(
            device=self.device, dtype=self.weight_dtype
        )
        keyboard_condition = cond_data["keyboard_condition"][:action_prefix_real_len].unsqueeze(0).to(
            device=self.device, dtype=self.weight_dtype
        )


        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype),
            "mouse_cond": mouse_condition,
            "keyboard_cond": keyboard_condition,
        }
        return conditional_dict


    def sample_timestep_tensor(self, num_latent_frames: int) -> Tuple[torch.Tensor, int, float]:
        """
        Sample a random diffusion timestep for the current 3-frame block
        """
        timesteps = self.scheduler.timesteps
        # t_idx = random.randint(0, len(timesteps) - 1)
        t_idx = random.randint(0,50)
        t_val = timesteps[t_idx].detach().float().item()

        timestep = timesteps[t_idx].to(device=self.device)
        timestep = timestep * torch.ones(
            [1, num_latent_frames],
            device=self.device,
            dtype=timesteps.dtype,
        )
        return timestep, t_idx, t_val

            
    def run_attack_forward_once(
        self,
        adv_first_frame: torch.Tensor,
    ):
        """
        Attack a local window:
            [1 proxy history frame] + [3 noisy target frames]

        - absolute time position is controlled by current_start
        - no cache
        - windowed_no_cache=True
        - only optimize target part (last 3 frames)
        """
        target_length = self.args.target_frames
        history_length = self.args.history_frames #1

        if target_length <= 0:
            raise ValueError("target_frames must be positive.")

        self.reset_model_block_masks()
        # block_idx=0: generate 0:2, block_idx=1: generate 3:5...
        block_idx = random.randint(0, self.args.max_block_idx)
        
        target_start_frame = block_idx * target_length
        window_start_frame = max(0, target_start_frame - history_length)
        actual_history_length = target_start_frame - window_start_frame
        total_length = actual_history_length + target_length   # = 4

        history_lat = self.clean_first_lat.repeat(1, 1, history_length, 1, 1)   # [1,16,1,44,80]

        noisy_target = torch.randn(
            [1, 16, target_length, 44, 80],
            device=self.device,
            dtype=self.weight_dtype,
        )

        if actual_history_length > 0:
            history_lat = self.clean_first_lat.repeat(1, 1, actual_history_length, 1, 1)
            x_input = torch.cat([history_lat, noisy_target], dim=2)
        else:
            x_input = noisy_target


        conditional_dict = self.build_conditional_dict(
            adv_first_frame=adv_first_frame,
            window_start_frame=window_start_frame,
            total_length=total_length,
            target_start_frame=target_start_frame,
            target_length=target_length,
        )

        target_timestep, t_idx, t_val = self.sample_timestep_tensor(target_length)
        timestep = torch.ones(
            [1, total_length],
            device=self.device,
            dtype=target_timestep.dtype,
        ) * target_timestep[:, :1]

        context_noise = float(getattr(self.config, "context_noise", 0))
        
        if actual_history_length > 0:
            timestep[:, :actual_history_length] = context_noise
        timestep[:, actual_history_length:] = target_timestep
        current_start = window_start_frame * self.frame_seq_length

        flow_pred, pred_x0 = self.generator(
            noisy_image_or_video=x_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            current_start=current_start,
            cache_start=current_start,
            windowed_no_cache=True,
        )
        flow_target = flow_pred[:, :, actual_history_length:]
        loss_flow = flow_target.float().pow(2).mean()
        aux_info = {
            "block_idx": block_idx,
            "timestep_index": t_idx,
        }

        return loss_flow, aux_info

    @torch.no_grad()
    def project_and_clamp(
        self,
        adv_x: torch.Tensor,
        original_x: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        eta = torch.clamp(adv_x - original_x, min=-eps, max=eps)
        adv_x = torch.clamp(original_x + eta, min=-1.0, max=1.0)
        return adv_x

    def attack(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        log_path = os.path.join(self.args.output_dir, "log.txt")
        log_f = open(log_path, "a", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_f)
        sys.stderr = Tee(sys.stderr, log_f)

        print("===== Run Args =====")
        print(json.dumps(vars(self.args), indent=2, ensure_ascii=False))
        print("====================")

        clean_first_frame = load_condition_image_tensor(
            self.args.input_image,
            device=self.device,
        )

        self.clean_first_frame = clean_first_frame.detach()
        self.clean_first_lat = self.encode_single_frame_to_latent(self.clean_first_frame).detach()

        max_total_length = max(
            max(0, block_idx * self.args.target_frames - self.args.history_frames) +
            (min(self.args.history_frames, block_idx * self.args.target_frames) + self.args.target_frames)
            for block_idx in range(self.args.max_block_idx + 1)
        )

        with torch.no_grad():
            self.clean_cond_concat_global = self.build_cond_concat(
                adv_first_frame=self.clean_first_frame,
                total_latent_frames=max_total_length,
            ).detach()

        print(f"[Check] clean_first_frame shape = {tuple(clean_first_frame.shape)}")
        print(
            f"[Check] clean_first_frame range = "
            f"({clean_first_frame.min().item():.4f}, {clean_first_frame.max().item():.4f})"
        )

        adv_first_frame = clean_first_frame.clone().detach()


        for step in range(1, self.args.num_steps + 1):
            adv_first_frame = adv_first_frame.detach().clone().requires_grad_(True)

            loss_flow, aux_info = self.run_attack_forward_once(
                adv_first_frame=adv_first_frame,
            )

            objective = loss_flow
            objective.backward()

            if adv_first_frame.grad is None:
                raise RuntimeError("adv_first_frame.grad is None. Check gradient flow.")

            with torch.no_grad():
                # Maximize objective: gradient ascent
                adv_first_frame = adv_first_frame + self.args.alpha * adv_first_frame.grad.sign()
                adv_first_frame = self.project_and_clamp(
                    adv_x=adv_first_frame,
                    original_x=clean_first_frame,
                    eps=self.args.eps,
                )

            obj_val = objective.detach().float().item()
            flow_val = loss_flow.detach().float().item()


            if step % self.args.log_every == 0 or step == 1:
                print(
                    f"[Step {step:04d}] "
                    f"obj={obj_val:.6f} | "
                    f"block={aux_info['block_idx']} | "
                    f"t_idx={aux_info['timestep_index']} | "
                )

            if step % self.args.save_every == 0 or step == self.args.num_steps:
                cur_path = os.path.join(self.args.output_dir, f"adv_step_{step:04d}.png")
                save_normalized_tensor_as_image(adv_first_frame, cur_path)

        final_path = os.path.join(self.args.output_dir, "adv_final.png")
        save_normalized_tensor_as_image(adv_first_frame, final_path)
        print(f"[Done] saved final adversarial frame to: {final_path}")

# ============================================================
# 3. CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        "PGD attack on Matrix-Game-2 input first frame (GTA only, first block only)"
    )

    # paths
    parser.add_argument("--input_image", type=str, default="demo_images/gta_drive/0000.png")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/inference_yaml/inference_gta_drive.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="Matrix-Game-2.0/gta_distilled_model/gta_keyboard2dim.safetensors",
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="Matrix-Game-2.0",
        help="Path containing Wan2.1_VAE.pth and related assets",
    )
    parser.add_argument("--output_dir", type=str, default="attack_outputs")

    # attack hyperparams
    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument(
        "--eps",
        type=float,
        default=0.05,
        help="L_inf budget in normalized [-1,1] domain",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.004,
        help="PGD step size in normalized [-1,1] domain",
    )

    # sequence setup
    parser.add_argument(
        "--target_frames",
        type=int,
        default=3,
        help="Only attack the first latent block; default is 3 frames.",
    )

    # optional override
    parser.add_argument(
        "--override_num_frame_per_block",
        type=int,
        default=None,
        help="Optional override for model.num_frame_per_block. Keep None to follow config/base model.",
    )

    parser.add_argument(
        "--history_frames",
        type=int,
        default=1,
        help="Proxy history latent frames before the target block. local_attn_size(4)-target block(3)=1."
    )

    parser.add_argument(
        "--action_history_latents",
        type=int,
        default=3,
        help="action_windows_size(3)-target(1)=2."
    )

    parser.add_argument(
        "--max_block_idx",
        type=int,
        default=4,
        help="Randomly sample target block index from [1, max_block_idx]. block_idx=1 means attacking frames 3:5."
    )

    # misc
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)

    return parser


def main():
    args = build_parser().parse_args()

    random.seed(args.seed)
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    attacker = FirstFramePGDAttack(args)
    attacker.attack()


if __name__ == "__main__":
    main()