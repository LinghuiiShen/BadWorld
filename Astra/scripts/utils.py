import torch
import torch.nn as nn
from einops import rearrange


def replace_dit_model_in_manager():
    """Replace DiT model class with MoE version."""
    from diffsynth.models.wan_video_dit_moe import WanModelMoe
    from diffsynth.configs.model_config import model_loader_configs

    for i, config in enumerate(model_loader_configs):
        keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource = config

        if "wan_video_dit" in model_names:
            new_model_names = []
            new_model_classes = []

            for name, cls in zip(model_names, model_classes):
                if name == "wan_video_dit":
                    new_model_names.append(name)
                    new_model_classes.append(WanModelMoe)
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)

            model_loader_configs[i] = (
                keys_hash,
                keys_hash_with_shape,
                new_model_names,
                new_model_classes,
                model_resource,
            )


def add_framepack_components(dit_model):
    """Add FramePack related components."""
    if hasattr(dit_model, "clean_x_embedder"):
        return

    inner_dim = dit_model.blocks[0].self_attn.q.weight.shape[0]

    class CleanXEmbedder(nn.Module):
        def __init__(self, inner_dim):
            super().__init__()
            self.proj = nn.Conv3d(
                16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)
            )
            self.proj_2x = nn.Conv3d(
                16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4)
            )
            self.proj_4x = nn.Conv3d(
                16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8)
            )

        def forward(self, x, scale="1x"):
            if scale == "1x":
                x = x.to(self.proj.weight.dtype)
                return self.proj(x)
            elif scale == "2x":
                x = x.to(self.proj_2x.weight.dtype)
                return self.proj_2x(x)
            elif scale == "4x":
                x = x.to(self.proj_4x.weight.dtype)
                return self.proj_4x(x)
            else:
                raise ValueError(f"Unsupported scale: {scale}")

    dit_model.clean_x_embedder = CleanXEmbedder(inner_dim)

    model_dtype = next(dit_model.parameters()).dtype
    dit_model.clean_x_embedder = dit_model.clean_x_embedder.to(dtype=model_dtype)


def add_moe_components(dit_model, moe_config):
    """Add MoE related components."""
    from diffsynth.models.wan_video_dit_moe import ModalityProcessor, MultiModalMoE

    if not hasattr(dit_model, "moe_config"):
        dit_model.moe_config = moe_config

    dit_model.top_k = moe_config.get("top_k", 1)

    dim = dit_model.blocks[0].self_attn.q.weight.shape[0]
    unified_dim = moe_config.get("unified_dim", 25)
    num_experts = moe_config.get("num_experts", 4)
    top_k = moe_config.get("top_k", 2)

    dit_model.sekai_processor = ModalityProcessor("sekai", 13, unified_dim)
    dit_model.nuscenes_processor = ModalityProcessor("nuscenes", 8, unified_dim)
    dit_model.openx_processor = ModalityProcessor("openx", 13, unified_dim)

    dit_model.global_router = nn.Linear(unified_dim, num_experts)

    for block in dit_model.blocks:
        block.moe = MultiModalMoE(
            unified_dim=unified_dim,
            output_dim=dim,
            num_experts=num_experts,
            top_k=top_k,
        )


def prepare_framepack_sliding_window_with_camera_moe(
    history_latents,
    target_frames_to_generate,
    camera_embedding_full,
    modality_type,
):
    """FramePack sliding window mechanism - MoE version.

    Args:
        history_latents: [C, T, H, W]
        target_frames_to_generate: number of new latent frames to generate
        camera_embedding_full: full camera embedding sequence
        modality_type: e.g. "sekai", "nuscenes", "openx"

    Returns:
        dict containing FramePack latents, indices, camera embedding and modality info
    """
    C, T, H, W = history_latents.shape

    total_indices_length = 1 + 16 + 2 + 1 + target_frames_to_generate

    start_frame = max(T - 20, 0)
    indices = torch.arange(
        start_frame,
        start_frame + total_indices_length,
        device=history_latents.device,
    )

    available_frames = min(T, 20)
    start_pos = 20 - available_frames

    indices[0] = indices[start_pos]

    if start_pos > 1:
        indices[1:start_pos] = -1

    split_sizes = [1, 16, 2, 1, target_frames_to_generate]

    (
        clean_latent_indices_start,
        clean_latent_4x_indices,
        clean_latent_2x_indices,
        clean_latent_1x_indices,
        latent_indices,
    ) = indices.split(split_sizes, dim=0)

    clean_latent_indices = torch.cat(
        [clean_latent_indices_start, clean_latent_1x_indices],
        dim=0,
    )

    clean_latents_combined = torch.zeros(
        C,
        20,
        H,
        W,
        dtype=history_latents.dtype,
        device=history_latents.device,
    )

    clean_latents_combined[:, start_pos:, :, :] = history_latents[
        :, -available_frames:, :, :
    ]

    start_latent = clean_latents_combined[:, start_pos:start_pos + 1, :, :]
    clean_latents_4x = clean_latents_combined[:, 1:17, :, :]
    clean_latents_2x = clean_latents_combined[:, 17:19, :, :]
    clean_latents_1x = clean_latents_combined[:, 19:20, :, :]

    clean_latents = torch.cat([start_latent, clean_latents_1x], dim=1)

    actual_needed_frames = T + target_frames_to_generate

    if camera_embedding_full.shape[0] < actual_needed_frames:
        shortage = actual_needed_frames - camera_embedding_full.shape[0]

        zero_motions = torch.eye(
            3,
            4,
            dtype=camera_embedding_full.dtype,
            device=camera_embedding_full.device,
        ).unsqueeze(0).repeat(shortage, 1, 1)

        padding = rearrange(zero_motions, "b c d -> b (c d)")

        # 注意：如果原 camera_embedding 有 mask 维度，即 shape[-1] = 13 或 8，
        # 这里需要给 padding 补上最后一维 mask。
        if padding.shape[1] < camera_embedding_full.shape[1]:
            pad_mask = torch.zeros(
                shortage,
                camera_embedding_full.shape[1] - padding.shape[1],
                dtype=camera_embedding_full.dtype,
                device=camera_embedding_full.device,
            )
            padding = torch.cat([padding, pad_mask], dim=1)

        camera_embedding_full = torch.cat([camera_embedding_full, padding], dim=0)

    combined_camera = camera_embedding_full[:actual_needed_frames, :].clone()

    # 最后一维作为 mask：1 表示 condition/history，0 表示 target
    combined_camera[:, -1] = 0.0
    combined_camera[:T, -1] = 1.0

    return {
        "latent_indices": latent_indices,
        "clean_latents": clean_latents,
        "clean_latents_2x": clean_latents_2x,
        "clean_latents_4x": clean_latents_4x,
        "clean_latent_indices": clean_latent_indices,
        "clean_latent_2x_indices": clean_latent_2x_indices,
        "clean_latent_4x_indices": clean_latent_4x_indices,
        "camera_embedding": combined_camera,
        "modality_type": modality_type,
        "current_length": T,
        "next_length": T + target_frames_to_generate,
    }