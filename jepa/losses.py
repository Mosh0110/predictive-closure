from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

def _require_latent_sequence(
    name: str, values: torch.Tensor, expected_time_steps: int
) -> None:
    if values.ndim != 3:
        raise ValueError(
            f"{name} must have shape (time, batch, latent), got {tuple(values.shape)}"
        )
    if values.shape[0] < 1 or values.shape[1] < 2 or values.shape[2] < 1:
        raise ValueError(
            f"{name} needs time >= 1, batch >= 2, and latent >= 1, "
            f"got {tuple(values.shape)}"
        )
    if values.shape[0] != expected_time_steps:
        raise ValueError(
            f"{name} must be time-major with {expected_time_steps} time steps, "
            f"got {tuple(values.shape)}"
        )
    if not values.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {values.dtype}")
    if not torch.isfinite(values).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def prediction_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    expected_time_steps: int = 16,
) -> torch.Tensor:
    if type(expected_time_steps) is not int or expected_time_steps <= 0:
        raise ValueError(
            "expected_time_steps must be a positive integer, "
            f"got {expected_time_steps!r}"
        )
    _require_latent_sequence("predictions", predictions, expected_time_steps)
    _require_latent_sequence("targets", targets, expected_time_steps)
    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions and targets must have identical shapes, got "
            f"{tuple(predictions.shape)} and {tuple(targets.shape)}"
        )
    return (predictions - targets).square().mean()


class SIGReg(nn.Module):
    def __init__(
        self,
        knots: int = 17,
        num_projections: int = 1024,
        expected_time_steps: int = 16,
    ) -> None:
        super().__init__()
        if type(knots) is not int or knots < 2:
            raise ValueError(f"knots must be an integer >= 2, got {knots!r}")
        if type(num_projections) is not int or num_projections <= 0:
            raise ValueError(
                "num_projections must be a positive integer, "
                f"got {num_projections!r}"
            )
        if type(expected_time_steps) is not int or expected_time_steps <= 0:
            raise ValueError(
                "expected_time_steps must be a positive integer, "
                f"got {expected_time_steps!r}"
            )
        self.knots = knots
        self.num_projections = num_projections
        self.expected_time_steps = expected_time_steps

        frequencies = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        spacing = 3.0 / (knots - 1)
        trapezoid_weights = torch.full(
            (knots,), 2.0 * spacing, dtype=torch.float32
        )
        trapezoid_weights[[0, -1]] = spacing
        gaussian_characteristic_function = torch.exp(-frequencies.square() / 2.0)
        self.register_buffer("frequencies", frequencies)
        self.register_buffer(
            "target_characteristic_function", gaussian_characteristic_function
        )
        self.register_buffer(
            "integration_weights",
            trapezoid_weights * gaussian_characteristic_function,
        )

    def _sample_directions(self, latents: torch.Tensor) -> torch.Tensor:
        latent_dim = latents.shape[2]
        if latent_dim == 1:
            directions = torch.randint(
                low=0,
                high=2,
                size=(1, self.num_projections),
                device=latents.device,
            ).to(dtype=latents.dtype)
            return directions.mul(2.0).sub(1.0)

        directions = torch.randn(
            latent_dim,
            self.num_projections,
            device=latents.device,
            dtype=latents.dtype,
        )
        norms = directions.norm(p=2, dim=0)
        if (norms == 0).any():
            raise FloatingPointError("SIGReg sampled a zero projection direction")
        return directions / norms

    def _loss_from_moments(
        self,
        cosine_mean: torch.Tensor,
        sine_mean: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        target = self.target_characteristic_function.to(dtype=cosine_mean.dtype)
        weights = self.integration_weights.to(dtype=cosine_mean.dtype)
        error = (cosine_mean - target).square() + sine_mean.square()
        per_projection = (error @ weights) * batch_size
        return per_projection.mean()

    def _moments(
        self, latents: torch.Tensor, directions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = self.frequencies.to(dtype=latents.dtype)
        angles = (latents @ directions).unsqueeze(-1) * frequencies
        return angles.cos().sum(dim=1), angles.sin().sum(dim=1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        _require_latent_sequence(
            "latents", latents, expected_time_steps=self.expected_time_steps
        )
        directions = self._sample_directions(latents)
        cosine_sum, sine_sum = self._moments(latents, directions)
        batch_size = latents.shape[1]
        return self._loss_from_moments(
            cosine_sum / batch_size,
            sine_sum / batch_size,
            batch_size,
        )

    def forward_chunked(self, latents: torch.Tensor, chunk_size: int) -> torch.Tensor:
        _require_latent_sequence(
            "latents", latents, expected_time_steps=self.expected_time_steps
        )
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be a positive integer, got {chunk_size!r}"
            )

        directions = self._sample_directions(latents)
        cosine_sum: torch.Tensor | None = None
        sine_sum: torch.Tensor | None = None
        for chunk in latents.split(chunk_size, dim=1):
            chunk_cosine_sum, chunk_sine_sum = self._moments(chunk, directions)
            cosine_sum = (
                chunk_cosine_sum
                if cosine_sum is None
                else cosine_sum + chunk_cosine_sum
            )
            sine_sum = (
                chunk_sine_sum if sine_sum is None else sine_sum + chunk_sine_sum
            )
        if cosine_sum is None or sine_sum is None:
            raise AssertionError("validated nonempty batch produced no chunks")

        batch_size = latents.shape[1]
        return self._loss_from_moments(
            cosine_sum / batch_size,
            sine_sum / batch_size,
            batch_size,
        )


def _flatten_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim == 2:
        flattened = latents
    elif latents.ndim == 3:
        flattened = latents.reshape(-1, latents.shape[-1])
    else:
        raise ValueError(
            "latents must have shape (samples, latent) or "
            f"(time, batch, latent), got {tuple(latents.shape)}"
        )
    if flattened.shape[0] < 2 or flattened.shape[1] < 1:
        raise ValueError(
            f"latents need at least two samples and one dimension, got {tuple(latents.shape)}"
        )
    if not flattened.is_floating_point():
        raise TypeError(f"latents must be floating point, got {flattened.dtype}")
    if not torch.isfinite(flattened).all():
        raise FloatingPointError("latents contain non-finite values")
    return flattened


def latent_effective_rank(latents: torch.Tensor) -> torch.Tensor:
    flattened = _flatten_latents(latents)
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (flattened.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    negative_tolerance = (
        max(covariance.shape)
        * torch.finfo(covariance.dtype).eps
        * covariance.abs().max().clamp_min(1.0)
    )
    if eigenvalues.min() < -negative_tolerance:
        raise FloatingPointError("computed latent covariance is not positive semidefinite")
    eigenvalues = eigenvalues.clamp_min(0.0)
    total = eigenvalues.sum()
    if total <= torch.finfo(total.dtype).eps:
        return total.new_zeros(())
    probabilities = eigenvalues / total
    positive = probabilities > 0.0
    entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
    return entropy.exp()


@dataclass(frozen=True)
class LatentCollapseMetrics:
    mean_variance: float
    minimum_variance: float
    maximum_variance: float
    effective_rank: float


def latent_collapse_metrics(latents: torch.Tensor) -> LatentCollapseMetrics:
    flattened = _flatten_latents(latents)
    per_dimension_variance = flattened.var(dim=0, unbiased=True)
    return LatentCollapseMetrics(
        mean_variance=float(per_dimension_variance.mean().item()),
        minimum_variance=float(per_dimension_variance.min().item()),
        maximum_variance=float(per_dimension_variance.max().item()),
        effective_rank=float(latent_effective_rank(flattened).item()),
    )

