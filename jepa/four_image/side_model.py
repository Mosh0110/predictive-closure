from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .models import CollageConvEncoder, FourImageJEPA


def current_c_crops(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 5 or tuple(frames.shape[2:]) != (3, 64, 64):
        raise ValueError("expected unmasked columns-layout collages [batch,time,3,64,64]")

    return frames[:, :-1, :, :32, 32:]


class VisibleCJEPA(FourImageJEPA):
    def __init__(self, *, side_seed: int, **kwargs) -> None:
        super().__init__(**kwargs)
        hidden = kwargs["predictor_hidden_dim"]

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(side_seed)
            self.side_encoder = CollageConvEncoder(
                self.latent_dim,
                base_channels=kwargs["base_channels"],
                representation_width=kwargs["representation_width"],
            )
            first = nn.Linear(2 * self.latent_dim, hidden)

        with torch.no_grad():
            first.weight[:, :self.latent_dim].copy_(self.predictor.network[0].weight)
            first.bias.copy_(self.predictor.network[0].bias)
        self.predictor.network[0] = first

    def side_latents(self, frames: torch.Tensor) -> torch.Tensor:
        crops = current_c_crops(frames)
        batch, times = crops.shape[:2]
        flat = crops.reshape(-1, 3, 32, 32)
        encoded = []
        for chunk in flat.split(self.encoder_frame_chunk_size):
            resized = F.interpolate(chunk, size=(64, 64), mode="bilinear", align_corners=False)
            encoded.append(self.side_encoder(resized))
        return torch.cat(encoded).reshape(batch, times, self.latent_dim).transpose(0, 1).contiguous()

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode_frames(frames)
        side = self.side_latents(frames)
        predictions = self.predictor.network(torch.cat((latents[:-1], side), dim=-1))
        return latents, predictions


def build_model(config) -> FourImageJEPA:
    kwargs = dict(
        sequence_length=config.feed.sequence_length,
        latent_dim=config.latent_dim,
        base_channels=config.encoder_base_channels,
        representation_width=config.encoder_representation_width,
        predictor_hidden_dim=config.predictor_hidden_dim,
        encoder_frame_chunk_size=config.encoder_frame_chunk_size,
    )
    if config.condition == "side":
        return VisibleCJEPA(side_seed=800000 + config.seed, **kwargs)
    return FourImageJEPA(**kwargs)
