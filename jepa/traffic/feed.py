from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import Tensor
from jepa.data import CIFAR10Pool
from jepa.four_image.feed import select_identity_pool


IMAGE_POSITIONS = ("A", "B")
IMAGE_POSITION_COUNT = len(IMAGE_POSITIONS)
COLOR_POSITION = "D"
COLOR_COUNT = 2
FRAME_SHAPE = (3, 64, 64)


def _probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0,1]")


@dataclass(frozen=True)
class ReplacementFeedConfig:
    sequence_length: int = 8
    visible_mode_probability: float = 1.0
    identity_count: int = 5_000
    identity_subset_seed: int = 20260805
    light_patch_size: int = 32
    light_strength: float = 1.0

    def __post_init__(self) -> None:
        if type(self.sequence_length) is not int or self.sequence_length < 2:
            raise ValueError("sequence_length must be an integer of at least two")
        _probability("visible_mode_probability", self.visible_mode_probability)
        if (
            type(self.identity_count) is not int
            or self.identity_count < 20
            or self.identity_count % 10
        ):
            raise ValueError("identity_count must be divisible by ten and at least 20")
        if (
            type(self.identity_subset_seed) is not int
            or not 0 <= self.identity_subset_seed < 2**63
        ):
            raise ValueError("identity_subset_seed must be an integer in [0,2^63)")
        if type(self.light_patch_size) is not int or not 1 <= self.light_patch_size <= 32:
            raise ValueError("light_patch_size must be an integer in [1,32]")
        _probability("light_strength", self.light_strength)
        if self.light_strength == 0.0:
            raise ValueError("light_strength must be positive")


@dataclass(frozen=True)
class ReplacementBatch:
    frames: Tensor
    image_states: Tensor
    image_labels: Tensor
    lights: Tensor
    modes: Tensor
    hidden_modes: Tensor
    use_visible_mode: Tensor


def color_palette() -> Tensor:
    palette = torch.tensor(
        [
            [0.08, 0.80, 0.12],
            [0.90, 0.08, 0.08],
        ],
        dtype=torch.float32,
    )
    if palette.shape != (COLOR_COUNT, 3):
        raise AssertionError("palette has the wrong shape")
    return palette


def _sample_different_identity(
    current: Tensor,
    identity_count: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    proposals = torch.randint(
        identity_count - 1,
        current.shape,
        generator=generator,
        dtype=torch.int64,
    )
    replacement = proposals + (proposals >= current).to(torch.int64)
    if bool((replacement == current).any()):
        raise AssertionError("replacement sampling retained an identity")
    return replacement


def sample_state_sequences(
    batch_size: int,
    identity_count: int,
    config: ReplacementFeedConfig,
    *,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if type(identity_count) is not int or identity_count < 2:
        raise ValueError("identity_count must be at least two")
    if generator.device.type != "cpu":
        raise ValueError("the feed generator must remain on CPU")

    lights = torch.randint(
        COLOR_COUNT,
        (batch_size, config.sequence_length),
        generator=generator,
        dtype=torch.int64,
    )
    states = [
        torch.randint(
            identity_count,
            (batch_size, IMAGE_POSITION_COUNT),
            generator=generator,
            dtype=torch.int64,
        )
    ]
    modes = []
    hidden_modes = []
    visible_branches = []
    row_indices = torch.arange(batch_size, dtype=torch.int64)
    for time_index in range(config.sequence_length - 1):
        hidden = torch.randint(
            COLOR_COUNT,
            (batch_size,),
            generator=generator,
            dtype=torch.int64,
        )
        use_visible = (
            torch.rand(batch_size, generator=generator)
            < config.visible_mode_probability
        )
        mode = torch.where(use_visible, lights[:, time_index], hidden)
        previous = states[-1]
        selected_current = previous[row_indices, mode]
        selected_replacement = _sample_different_identity(
            selected_current,
            identity_count,
            generator=generator,
        )
        following = previous.clone()
        following[row_indices, mode] = selected_replacement
        changed = following != previous
        if not torch.equal(changed.sum(dim=1), torch.ones(batch_size, dtype=torch.int64)):
            raise AssertionError("each transition must replace exactly one image")
        if not bool(changed[row_indices, mode].all()):
            raise AssertionError("the selected position was not replaced")
        states.append(following)
        modes.append(mode)
        hidden_modes.append(hidden)
        visible_branches.append(use_visible)
    return (
        torch.stack(states, dim=1),
        lights,
        torch.stack(modes, dim=1),
        torch.stack(hidden_modes, dim=1),
        torch.stack(visible_branches, dim=1),
    )


def render_collages(
    images: Tensor,
    lights: Tensor,
    palette: Tensor,
    *,
    light_patch_size: int = 32,
) -> Tensor:
    if images.ndim != 6 or tuple(images.shape[2:]) != (IMAGE_POSITION_COUNT, 3, 32, 32):
        raise ValueError("images must have shape [batch,time,2,3,32,32]")
    if lights.shape != images.shape[:2] or lights.dtype != torch.int64:
        raise ValueError("lights must have shape [batch,time] and dtype int64")
    if palette.shape != (COLOR_COUNT, 3) or not palette.is_floating_point():
        raise ValueError("palette must have shape [2,3]")
    if images.device != lights.device or images.device != palette.device:
        raise ValueError("images, lights, and palette must share a device")
    if not images.is_floating_point():
        raise TypeError("images must be floating point")
    if type(light_patch_size) is not int or not 1 <= light_patch_size <= 32:
        raise ValueError("light_patch_size must be an integer in [1,32]")

    batch_size, time_steps = images.shape[:2]
    frames = torch.empty(
        batch_size,
        time_steps,
        *FRAME_SHAPE,
        dtype=images.dtype,
        device=images.device,
    )
    frames[:, :, :, :32, :32] = images[:, :, 0]
    frames[:, :, :, :32, 32:] = images[:, :, 1]
    frames[:, :, :, 32:, :32] = 0.0
    frames[:, :, :, 32:, 32:] = 0.0
    light_pixels = palette.index_select(0, lights.reshape(-1)).reshape(
        batch_size, time_steps, 3, 1, 1
    )
    offset = 32 + (32 - light_patch_size) // 2
    end = offset + light_patch_size
    frames[:, :, :, offset:end, offset:end] = light_pixels
    return frames


def _bayes_accuracy(source: Tensor, target: Tensor) -> float:
    if source.shape != target.shape or source.ndim != 1:
        raise ValueError("source and target must be equal one-dimensional tensors")
    counts = torch.bincount(
        source * COLOR_COUNT + target,
        minlength=COLOR_COUNT * COLOR_COUNT,
    ).reshape(COLOR_COUNT, COLOR_COUNT)
    return float(counts.max(dim=1).values.sum() / counts.sum())


def batch_statistics(batch: ReplacementBatch) -> dict[str, object]:
    current_lights = batch.lights[:, :-1].reshape(-1).cpu()
    future_lights = batch.lights[:, 1:].reshape(-1).cpu()
    modes = batch.modes.reshape(-1).cpu()
    changed = batch.image_states[:, 1:] != batch.image_states[:, :-1]
    per_position_persistence = {
        position: float((~changed[:, :, index]).float().mean())
        for index, position in enumerate(IMAGE_POSITIONS)
    }
    return {
        "visible_mode_probability": batch.use_visible_mode.float().mean().item(),
        "visible_light_matches_mode": float(
            (batch.lights[:, :-1] == batch.modes).float().mean()
        ),
        "next_light_bayes_accuracy_from_current_light": _bayes_accuracy(
            current_lights, future_lights
        ),
        "mode_bayes_accuracy_from_visible_light": _bayes_accuracy(
            current_lights, modes
        ),
        "changed_positions_per_transition": float(changed.sum(dim=2).float().mean()),
        "per_position_persistence": per_position_persistence,
        "light_histogram": torch.bincount(
            batch.lights.reshape(-1).cpu(), minlength=COLOR_COUNT
        ).tolist(),
        "mode_histogram": torch.bincount(modes, minlength=COLOR_COUNT).tolist(),
    }


class ReplacementSubsidyFeed:
    def __init__(self, pool: CIFAR10Pool, config: ReplacementFeedConfig) -> None:
        if pool.images.shape[0] != config.identity_count:
            raise ValueError("pool identity count differs from the feed configuration")
        self.pool = pool
        self.config = config
        self.palette = color_palette().mul(config.light_strength)

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: str | torch.device,
    ) -> ReplacementBatch:
        states, lights, modes, hidden_modes, use_visible = sample_state_sequences(
            batch_size,
            self.pool.images.shape[0],
            self.config,
            generator=generator,
        )
        target = torch.device(device)
        rendered_images = self.pool.images[states].to(
            target, dtype=torch.float32
        ).div_(255.0)
        target_lights = lights.to(target)
        frames = render_collages(
            rendered_images,
            target_lights,
            self.palette.to(target),
            light_patch_size=self.config.light_patch_size,
        )
        expected = (batch_size, self.config.sequence_length, *FRAME_SHAPE)
        if tuple(frames.shape) != expected:
            raise AssertionError("renderer produced the wrong shape")
        target_states = states.to(target)
        return ReplacementBatch(
            frames=frames,
            image_states=target_states,
            image_labels=self.pool.labels.to(target).index_select(
                0, target_states.reshape(-1)
            ).reshape_as(target_states),
            lights=target_lights,
            modes=modes.to(target),
            hidden_modes=hidden_modes.to(target),
            use_visible_mode=use_visible.to(target),
        )


__all__ = [
    "COLOR_COUNT",
    "COLOR_POSITION",
    "FRAME_SHAPE",
    "IMAGE_POSITIONS",
    "ReplacementBatch",
    "ReplacementFeedConfig",
    "ReplacementSubsidyFeed",
    "batch_statistics",
    "color_palette",
    "render_collages",
    "sample_state_sequences",
    "select_identity_pool",
]
