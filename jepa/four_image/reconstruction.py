from __future__ import annotations
import hashlib
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor, nn


RECONSTRUCTION_POSITIONS = (1,)
RECONSTRUCTION_POSITION_NAMES = ("C",)
COARSE_IMAGE_SIZE = 4
CHANNEL_COUNT = 3


def _require_pool_images(images: Tensor) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
        raise ValueError("pool images must have shape [identity,3,32,32]")
    if not images.is_floating_point():
        raise TypeError("pool images must be floating point")
    if not torch.isfinite(images).all():
        raise FloatingPointError("pool images contain non-finite values")
    if float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise ValueError("pool images must lie in [0,1]")


def coarse_pool_images(images: Tensor, size: int = COARSE_IMAGE_SIZE) -> Tensor:
    _require_pool_images(images)
    if type(size) is not int or not 1 <= size <= 32 or 32 % size:
        raise ValueError("size must be a positive divisor of 32")
    coarse = F.avg_pool2d(images, kernel_size=32 // size, stride=32 // size)
    expected = (images.shape[0], CHANNEL_COUNT, size, size)
    if tuple(coarse.shape) != expected:
        raise AssertionError("coarse pooling produced the wrong shape")
    return coarse.flatten(start_dim=1).contiguous()


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    for field in (str(value.dtype), repr(tuple(value.shape))):
        encoded = field.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True)
class CoarseTargetStandardizer:
    mean: Tensor
    std: Tensor
    size: int

    @classmethod
    def fit(cls, coarse_images: Tensor, size: int) -> "CoarseTargetStandardizer":
        expected_width = CHANNEL_COUNT * size * size
        if coarse_images.ndim != 2 or coarse_images.shape[1] != expected_width:
            raise ValueError("coarse images have the wrong shape")
        if coarse_images.shape[0] < 2:
            raise ValueError("standardization requires at least two identities")
        mean = coarse_images.mean(dim=0)
        std = coarse_images.std(dim=0, unbiased=False)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise FloatingPointError("non-finite reconstruction standardization")
        if (std <= 1e-8).any():
            raise ValueError("a reconstruction target element has negligible variance")
        return cls(mean=mean, std=std, size=size)

    def contract(self) -> dict[str, object]:
        return {
            "fit_universe": "deterministic class-balanced JEPA training identity pool",
            "fit_labels_used": False,
            "mean_sha256": tensor_sha256(self.mean),
            "std_sha256": tensor_sha256(self.std),
            "minimum_std": float(self.std.min()),
            "maximum_std": float(self.std.max()),
        }


def standardized_reconstruction_targets(
    states: Tensor,
    coarse_images: Tensor,
    standardizer: CoarseTargetStandardizer,
    positions: tuple[int, ...] = RECONSTRUCTION_POSITIONS,
) -> Tensor:
    if states.ndim != 3 or states.shape[2] != 4:
        raise ValueError("states must have shape [batch,time,4]")
    if states.dtype != torch.int64:
        raise TypeError("states must be int64")
    if tuple(positions) != RECONSTRUCTION_POSITIONS:
        raise ValueError("this intervention reconstructs exactly logical B")
    expected_width = CHANNEL_COUNT * standardizer.size * standardizer.size
    if coarse_images.ndim != 2 or coarse_images.shape[1] != expected_width:
        raise ValueError("coarse image pool has the wrong shape")
    if states.device != coarse_images.device:
        raise ValueError("states and coarse images must share a device")
    if standardizer.mean.device != states.device or standardizer.std.device != states.device:
        raise ValueError("standardizer and states must share a device")
    logical = states[:, :, list(positions)]
    selected = coarse_images[logical]
    standardized = (selected - standardizer.mean) / standardizer.std
    batch, time, position_count, width = standardized.shape
    result = standardized.reshape(batch, time, position_count * width)
    result = result.transpose(0, 1).contiguous()
    expected = (states.shape[1], states.shape[0], len(positions) * expected_width)
    if tuple(result.shape) != expected:
        raise AssertionError("reconstruction targets have the wrong shape")
    if not torch.isfinite(result).all():
        raise FloatingPointError("reconstruction targets contain non-finite values")
    return result


class LinearCoarseReconstructionDecoder(nn.Module):
    def __init__(self, latent_dim: int, target_size: int = COARSE_IMAGE_SIZE) -> None:
        super().__init__()
        if type(latent_dim) is not int or latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if target_size != COARSE_IMAGE_SIZE:
            raise ValueError(f"target_size must equal {COARSE_IMAGE_SIZE}")
        self.latent_dim = latent_dim
        self.target_size = target_size
        self.output_dim = (
            len(RECONSTRUCTION_POSITIONS)
            * CHANNEL_COUNT
            * target_size
            * target_size
        )
        self.linear = nn.Linear(latent_dim, self.output_dim)

    def forward(self, latents: Tensor) -> Tensor:
        if latents.ndim != 3 or latents.shape[2] != self.latent_dim:
            raise ValueError("latents must have shape [time,batch,latent_dim]")
        return self.linear(latents)


def reconstruction_mse(predictions: Tensor, targets: Tensor) -> Tensor:
    if predictions.ndim != 3 or targets.ndim != 3:
        raise ValueError("reconstruction values must be time-major rank-three tensors")
    if predictions.shape != targets.shape:
        raise ValueError("reconstruction predictions and targets must have equal shapes")
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise TypeError("reconstruction values must be floating point")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise FloatingPointError("reconstruction values contain non-finite values")
    return (predictions - targets).square().mean()


__all__ = [
    "CHANNEL_COUNT",
    "COARSE_IMAGE_SIZE",
    "RECONSTRUCTION_POSITIONS",
    "RECONSTRUCTION_POSITION_NAMES",
    "CoarseTargetStandardizer",
    "LinearCoarseReconstructionDecoder",
    "coarse_pool_images",
    "reconstruction_mse",
    "standardized_reconstruction_targets",
    "tensor_sha256",
]
