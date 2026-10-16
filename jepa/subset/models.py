from __future__ import annotations
import torch
from torch import nn
from .feed import FRAME_SHAPE


class RingCollageConvEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        *,
        base_channels: int = 32,
        representation_width: int = 1024,
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("base_channels", base_channels),
            ("representation_width", representation_width),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, 2 * base_channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * base_channels, 4 * base_channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(4 * base_channels, 8 * base_channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.AdaptiveMaxPool2d((4, 4)),
        )
        self.projection = nn.Sequential(
            nn.Linear(8 * base_channels * 4 * 4, representation_width),
            nn.GELU(),
            nn.Linear(representation_width, latent_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or tuple(frames.shape[1:]) != FRAME_SHAPE:
            raise ValueError(f"frames must have shape [batch,{','.join(map(str, FRAME_SHAPE))}]")
        features = self.features(frames)
        expected = (frames.shape[0], 8 * self.base_channels, 4, 4)
        if tuple(features.shape) != expected:
            raise AssertionError("convolution produced the wrong feature shape")
        return self.projection(features.flatten(start_dim=1))


class LatentMLPPredictor(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        if type(latent_dim) is not int or latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[-1] != self.latent_dim:
            raise ValueError("latents have the wrong final dimension")
        return self.network(latents)


class RingBudgetJEPA(nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        latent_dim: int,
        base_channels: int,
        representation_width: int,
        predictor_hidden_dim: int,
        encoder_frame_chunk_size: int,
    ) -> None:
        super().__init__()
        if type(sequence_length) is not int or sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if type(encoder_frame_chunk_size) is not int or encoder_frame_chunk_size <= 0:
            raise ValueError("encoder_frame_chunk_size must be positive")
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.encoder_frame_chunk_size = encoder_frame_chunk_size
        self.encoder = RingCollageConvEncoder(
            latent_dim,
            base_channels=base_channels,
            representation_width=representation_width,
        )
        self.predictor = LatentMLPPredictor(latent_dim, predictor_hidden_dim)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        expected = (self.sequence_length, *FRAME_SHAPE)
        if frames.ndim != 5 or tuple(frames.shape[1:]) != expected:
            raise ValueError("frames have the wrong sequence shape")
        batch_size = frames.shape[0]
        flattened = frames.reshape(batch_size * self.sequence_length, *FRAME_SHAPE)
        encoded = torch.cat(
            [self.encoder(chunk) for chunk in flattened.split(self.encoder_frame_chunk_size)]
        )
        return encoded.reshape(
            batch_size, self.sequence_length, self.latent_dim
        ).transpose(0, 1).contiguous()

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode_frames(frames)
        return latents, self.predictor(latents[:-1])


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "LatentMLPPredictor",
    "RingBudgetJEPA",
    "RingCollageConvEncoder",
    "parameter_count",
]
