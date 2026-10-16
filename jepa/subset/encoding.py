from __future__ import annotations
import torch
from torch import Tensor, nn
from .feed import PositionImagePool, render_collages

@torch.no_grad()
def encode_states(
    encoder: nn.Module,
    pool: PositionImagePool,
    states: Tensor,
    *,
    ring_size: int,
    bd_render_tile_size: int | None,
    bd_layout: str,
    batch_size: int,
    device: torch.device,
    bf16: bool,
) -> Tensor:
    if states.ndim != 3 or states.shape[-1] != 2 + ring_size:
        raise ValueError("probe states have the wrong ring shape")
    flat = states.reshape(-1, states.shape[-1])
    results = []
    for start in range(0, flat.shape[0], batch_size):
        indices = flat[start : start + batch_size]
        images = pool.images[indices].to(device).float().div_(255.0)
        frames = render_collages(
            images[:, None],
            ring_size,
            bd_render_tile_size=bd_render_tile_size,
            bd_layout=bd_layout,
        )[:, 0]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            results.append(encoder(frames).float().cpu())
    return torch.cat(results).reshape(*states.shape[:2], -1)

