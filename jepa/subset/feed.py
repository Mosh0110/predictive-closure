from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass, fields
from typing import Any, Mapping
import torch
import torch.nn.functional as F
from torch import Tensor
from jepa.data import CIFAR10Pool, NUM_CLASSES, source_indices_sha256
from jepa.four_image.feed import select_identity_pool


FRAME_SHAPE = (3, 128, 256)
IDENTITY_COUNT = 5_000
VALID_BD_RING_SIZES = (2, 4, 8, 16)
BD_LAYOUTS = ("constant_area256",)
CONSTANT_AREA_TILE_SIZES = {2: 90, 4: 64, 8: 45, 16: 32}
CONSTANT_AREA_GRIDS = {2: (1, 2), 4: (2, 2), 8: (2, 4), 16: (4, 4)}
SHARED_DRAW_WIDTH = 6
MAX_POSITION_COUNT = 2 + max(VALID_BD_RING_SIZES)
AC_POSITION_NAMES = ("A", "B")
BD_POSITION_NAMES = ("C", "D", "E", "F", "G", "H", "I", "J",
                     "K", "L", "M", "N", "O", "P", "Q", "R")
AC_INDICES = (0, 1)


def _probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0,1]")


def position_names(bd_ring_size: int) -> tuple[str, ...]:
    if type(bd_ring_size) is not int or bd_ring_size not in VALID_BD_RING_SIZES:
        raise ValueError(f"bd_ring_size must be one of {VALID_BD_RING_SIZES}")
    return AC_POSITION_NAMES + BD_POSITION_NAMES[:bd_ring_size]


def source_position_for_target(bd_ring_size: int) -> tuple[int, ...]:
    position_names(bd_ring_size)
    bd_indices = tuple(range(2, 2 + bd_ring_size))
    return (1, 0, bd_indices[-1], *bd_indices[:-1])


def bd_tile_size(bd_ring_size: int, override: int | None = None) -> int:
    position_names(bd_ring_size)
    default = CONSTANT_AREA_TILE_SIZES[bd_ring_size]
    if override is None:
        return default
    if type(override) is not int:
        raise TypeError("bd_render_tile_size must be an integer or None")
    if override != default:
        raise ValueError(f"N={bd_ring_size} requires tile size {default}")
    return override


def bd_tile_coordinates_yx(
    bd_ring_size: int,
    override: int | None = None,
    *,
    layout: str = "constant_area256",
) -> tuple[tuple[int, int], ...]:
    tile_size = bd_tile_size(bd_ring_size, override)
    if layout not in BD_LAYOUTS:
        raise ValueError(f"bd_layout must be one of {BD_LAYOUTS}")
    rows, columns = CONSTANT_AREA_GRIDS[bd_ring_size]
    top = (128 - rows * tile_size) // 2
    left = 64 + (192 - columns * tile_size) // 2
    return tuple((top + (i // columns) * tile_size, left + (i % columns) * tile_size)
                 for i in range(bd_ring_size))


@dataclass(frozen=True)
class RingBudgetFeedConfig:
    sequence_length: int = 8
    bd_ring_size: int = 2
    q_ac: float = 0.9
    q_bd: float = 0.95
    identity_subset_seed: int = 20260805
    bd_render_tile_size: int | None = None
    bd_copy_probabilities: tuple[float, ...] | None = None
    bd_layout: str = "constant_area256"

    def __post_init__(self) -> None:
        if type(self.sequence_length) is not int or self.sequence_length < 2:
            raise ValueError("sequence_length must be an integer of at least two")
        position_names(self.bd_ring_size)
        _probability("q_ac", self.q_ac)
        _probability("q_bd", self.q_bd)
        if self.bd_copy_probabilities is not None:
            if type(self.bd_copy_probabilities) is not tuple:
                raise TypeError("bd_copy_probabilities must be a tuple or None")
            if len(self.bd_copy_probabilities) != self.bd_ring_size:
                raise ValueError(
                    "bd_copy_probabilities must have exactly bd_ring_size entries"
                )
            for index, probability in enumerate(self.bd_copy_probabilities):
                _probability(f"bd_copy_probabilities[{index}]", probability)
            configured_mean = sum(self.bd_copy_probabilities) / self.bd_ring_size
            if not math.isclose(configured_mean, self.q_bd, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "q_bd must equal the exact mean of bd_copy_probabilities"
                )
        if (
            type(self.identity_subset_seed) is not int
            or not 0 <= self.identity_subset_seed < 2**63
        ):
            raise ValueError("identity_subset_seed must be an integer in [0,2^63)")
        bd_tile_coordinates_yx(
            self.bd_ring_size,
            self.bd_render_tile_size,
            layout=self.bd_layout,
        )

    @property
    def position_names(self) -> tuple[str, ...]:
        return position_names(self.bd_ring_size)

    @property
    def position_count(self) -> int:
        return 2 + self.bd_ring_size

    @property
    def ac_indices(self) -> tuple[int, int]:
        return AC_INDICES

    @property
    def bd_indices(self) -> tuple[int, ...]:
        return tuple(range(2, self.position_count))

    @property
    def source_positions(self) -> tuple[int, ...]:
        return source_position_for_target(self.bd_ring_size)

    @property
    def probabilities(self) -> tuple[float, ...]:
        return (self.q_ac, self.q_ac, *self.bd_probabilities)

    @property
    def bd_probabilities(self) -> tuple[float, ...]:
        if self.bd_copy_probabilities is None:
            return tuple(self.q_bd for _ in self.bd_indices)
        return self.bd_copy_probabilities

    @property
    def render_tile_size(self) -> int:
        return bd_tile_size(self.bd_ring_size, self.bd_render_tile_size)

    @property
    def render_coordinates_yx(self) -> tuple[tuple[int, int], ...]:
        return bd_tile_coordinates_yx(
            self.bd_ring_size,
            self.bd_render_tile_size,
            layout=self.bd_layout,
        )


def ring_budget_feed_config_from_mapping(
    value: Mapping[str, Any],
) -> RingBudgetFeedConfig:
    if not isinstance(value, Mapping):
        raise TypeError("ring-budget feed config must be an object")
    normalized = dict(value)
    normalized.setdefault("bd_copy_probabilities", None)
    normalized.setdefault("bd_layout", "constant_area256")
    expected = {field.name for field in fields(RingBudgetFeedConfig)}
    missing = sorted(expected - set(normalized))
    extra = sorted(set(normalized) - expected)
    if missing or extra:
        raise ValueError(
            "ring-budget feed config keys differ from schema; "
            f"missing={missing}, extra={extra}"
        )
    probabilities = normalized["bd_copy_probabilities"]
    if isinstance(probabilities, list):
        normalized["bd_copy_probabilities"] = tuple(probabilities)
    return RingBudgetFeedConfig(**normalized)


@dataclass(frozen=True)
class PositionImagePool:
    images: Tensor
    labels: Tensor
    source_indices: Tensor
    dataset_classes: tuple[int, ...] = tuple(range(NUM_CLASSES))

    def __post_init__(self) -> None:
        if self.images.device.type != "cpu" or self.images.dtype != torch.uint8:
            raise TypeError("pool images must be CPU uint8")
        if tuple(self.images.shape) != (IDENTITY_COUNT, 3, 32, 32):
            raise ValueError(
                f"pool images must have shape [{IDENTITY_COUNT},3,32,32]"
            )
        if self.labels.device.type != "cpu" or self.labels.dtype != torch.int64:
            raise TypeError("pool labels must be CPU int64")
        if tuple(self.labels.shape) != (IDENTITY_COUNT,):
            raise ValueError(f"pool labels must have shape [{IDENTITY_COUNT}]")
        if (
            self.source_indices.device.type != "cpu"
            or self.source_indices.dtype != torch.int64
        ):
            raise TypeError("source_indices must be CPU int64")
        if tuple(self.source_indices.shape) != (IDENTITY_COUNT,):
            raise ValueError(f"source_indices must have shape [{IDENTITY_COUNT}]")
        if self.source_indices.unique().numel() != IDENTITY_COUNT:
            raise ValueError("source_indices must be unique")
        if bool((self.source_indices < 0).any()):
            raise ValueError("source_indices must be nonnegative")
        if not bool((self.source_indices[1:] > self.source_indices[:-1]).all()):
            raise ValueError("source_indices must be strictly increasing")
        if self.dataset_classes != tuple(range(NUM_CLASSES)):
            raise ValueError("the ring pool must contain all ten CIFAR classes")
        histogram = torch.bincount(self.labels, minlength=NUM_CLASSES)
        expected = torch.full((NUM_CLASSES,), IDENTITY_COUNT // NUM_CLASSES)
        if not torch.equal(histogram, expected):
            raise ValueError(
                "the ring pool must contain exactly 500 images from each class"
            )

    @property
    def identity_count(self) -> int:
        return IDENTITY_COUNT


@dataclass(frozen=True)
class RingBudgetBatch:
    frames: Tensor
    states: Tensor
    labels: Tensor
    source_image_indices: Tensor
    copy_branch: Tensor
    random_proposals: Tensor
    copied_targets: Tensor


def select_ring_pool(
    full_pool: CIFAR10Pool,
    config: RingBudgetFeedConfig,
) -> PositionImagePool:
    if full_pool.split != "train":
        raise ValueError("the ring pool must be selected from CIFAR-10 train")
    selected, source_indices = select_identity_pool(
        full_pool,
        identity_count=IDENTITY_COUNT,
        seed=config.identity_subset_seed,
    )
    return PositionImagePool(
        images=selected.images,
        labels=selected.labels,
        source_indices=source_indices,
    )


def copied_targets_for_states(states: Tensor, bd_ring_size: int) -> Tensor:
    expected_positions = 2 + bd_ring_size
    position_names(bd_ring_size)
    if states.ndim != 3 or states.shape[1] < 2 or states.shape[2] != expected_positions:
        raise ValueError(
            f"states must have shape [batch,time,{expected_positions}] with time at least two"
        )
    if states.dtype != torch.int64:
        raise TypeError("states must be int64")
    sources = torch.tensor(
        source_position_for_target(bd_ring_size),
        dtype=torch.int64,
        device=states.device,
    )
    return states[:, :-1].index_select(2, sources)


def sample_state_sequences(
    batch_size: int,
    config: RingBudgetFeedConfig,
    *,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if generator.device.type != "cpu":
        raise ValueError("the feed generator must remain on CPU")
    probabilities = torch.tensor(config.probabilities, dtype=torch.float32)
    sources = torch.tensor(config.source_positions, dtype=torch.int64)


    extension_generator = None
    extra_generator = None
    if config.position_count > SHARED_DRAW_WIDTH:
        state_bytes = generator.get_state().contiguous().numpy().tobytes()
        digest = hashlib.sha256(b"ring-budget-n8-extension-v1\0" + state_bytes)
        extension_seed = int.from_bytes(digest.digest()[:8], "little") % (2**63)
        extension_generator = torch.Generator(device="cpu").manual_seed(
            extension_seed
        )
        if config.position_count > 10:
            extra_digest = hashlib.sha256(b"ring-budget-n16-extension-v1\0" + state_bytes)
            extra_generator = torch.Generator(device="cpu").manual_seed(
                int.from_bytes(extra_digest.digest()[:8], "little") % (2**63)
            )

    initial_prefix = torch.randint(
        IDENTITY_COUNT,
        (batch_size, SHARED_DRAW_WIDTH),
        generator=generator,
        dtype=torch.int64,
    )
    if extension_generator is None:
        initial = initial_prefix[:, : config.position_count]
    else:
        initial_extension = torch.randint(
            IDENTITY_COUNT,
            (batch_size, 10 - SHARED_DRAW_WIDTH),
            generator=extension_generator,
            dtype=torch.int64,
        )[:, : config.position_count - SHARED_DRAW_WIDTH]
        initial = torch.cat((initial_prefix, initial_extension), dim=1)
        if extra_generator is not None:
            initial_extra = torch.randint(
                IDENTITY_COUNT, (batch_size, MAX_POSITION_COUNT - 10),
                generator=extra_generator, dtype=torch.int64,
            )
            initial = torch.cat((initial, initial_extra), dim=1)
    states = [initial]
    branches = []
    random_proposals = []
    copied_targets = []
    for _ in range(config.sequence_length - 1):
        copied = states[-1].index_select(1, sources)
        branch_prefix = torch.rand(
            batch_size,
            SHARED_DRAW_WIDTH,
            generator=generator,
        )
        random_prefix = torch.randint(
            IDENTITY_COUNT,
            (batch_size, SHARED_DRAW_WIDTH),
            generator=generator,
            dtype=torch.int64,
        )
        if extension_generator is None:
            branch_uniform = branch_prefix[:, : config.position_count]
            random = random_prefix[:, : config.position_count]
        else:
            branch_extension = torch.rand(
                batch_size,
                10 - SHARED_DRAW_WIDTH,
                generator=extension_generator,
            )[:, : config.position_count - SHARED_DRAW_WIDTH]
            random_extension = torch.randint(
                IDENTITY_COUNT,
                (batch_size, 10 - SHARED_DRAW_WIDTH),
                generator=extension_generator,
                dtype=torch.int64,
            )[:, : config.position_count - SHARED_DRAW_WIDTH]
            branch_uniform = torch.cat((branch_prefix, branch_extension), dim=1)
            random = torch.cat((random_prefix, random_extension), dim=1)
            if extra_generator is not None:
                branch_extra = torch.rand(
                    batch_size, MAX_POSITION_COUNT - 10, generator=extra_generator
                )
                random_extra = torch.randint(
                    IDENTITY_COUNT, (batch_size, MAX_POSITION_COUNT - 10),
                    generator=extra_generator, dtype=torch.int64,
                )
                branch_uniform = torch.cat((branch_uniform, branch_extra), dim=1)
                random = torch.cat((random, random_extra), dim=1)
        branch = branch_uniform < probabilities
        states.append(torch.where(branch, copied, random))
        branches.append(branch)
        random_proposals.append(random)
        copied_targets.append(copied)
    return (
        torch.stack(states, dim=1),
        torch.stack(branches, dim=1),
        torch.stack(random_proposals, dim=1),
        torch.stack(copied_targets, dim=1),
    )


def render_collages(
    images: Tensor,
    bd_ring_size: int,
    bd_render_tile_size: int | None = None,
    bd_layout: str = "constant_area256",
) -> Tensor:
    expected_positions = 2 + bd_ring_size
    position_names(bd_ring_size)
    if images.ndim != 6 or tuple(images.shape[2:]) != (
        expected_positions,
        3,
        32,
        32,
    ):
        raise ValueError(
            "images must have shape "
            f"[batch,time,{expected_positions},3,32,32]"
        )
    if not images.is_floating_point():
        raise TypeError("images must be floating point")
    tile_size = bd_tile_size(bd_ring_size, bd_render_tile_size)
    batch_size, time_steps = images.shape[:2]
    frames = torch.zeros(
        batch_size,
        time_steps,
        *FRAME_SHAPE,
        dtype=images.dtype,
        device=images.device,
    )

    frames[:, :, :, 16:48, 16:48] = images[:, :, 0]
    frames[:, :, :, 80:112, 16:48] = images[:, :, 1]

    bd = images[:, :, 2:].reshape(-1, 3, 32, 32)
    if tile_size != 32:
        bd = F.interpolate(
            bd,
            size=(tile_size, tile_size),
            mode="bilinear",
            align_corners=False,
        )
    bd = bd.reshape(
        batch_size,
        time_steps,
        bd_ring_size,
        3,
        tile_size,
        tile_size,
    )
    coordinates = bd_tile_coordinates_yx(
        bd_ring_size,
        bd_render_tile_size,
        layout=bd_layout,
    )
    for ring_index, (y_start, x_start) in enumerate(coordinates):
        frames[
            :,
            :,
            :,
            y_start : y_start + tile_size,
            x_start : x_start + tile_size,
        ] = bd[:, :, ring_index]
    return frames


class RingBudgetFeed:
    def __init__(self, pool: PositionImagePool, config: RingBudgetFeedConfig) -> None:
        if not isinstance(pool, PositionImagePool):
            raise TypeError("pool must be a ring_budget_switch PositionImagePool")
        self.pool = pool
        self.config = config

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: str | torch.device,
    ) -> RingBudgetBatch:
        states, branches, random, copied = sample_state_sequences(
            batch_size,
            self.config,
            generator=generator,
        )
        target = torch.device(device)
        images = self.pool.images[states].to(target, dtype=torch.float32).div_(255.0)
        frames = render_collages(
            images,
            bd_ring_size=self.config.bd_ring_size,
            bd_render_tile_size=self.config.bd_render_tile_size,
            bd_layout=self.config.bd_layout,
        )
        expected_shape = (
            batch_size,
            self.config.sequence_length,
            *FRAME_SHAPE,
        )
        if tuple(frames.shape) != expected_shape:
            raise AssertionError("renderer produced the wrong frame shape")
        return RingBudgetBatch(
            frames=frames,
            states=states.to(target),
            labels=self.pool.labels[states].to(target),
            source_image_indices=self.pool.source_indices[states].to(target),
            copy_branch=branches.to(target),
            random_proposals=random.to(target),
            copied_targets=copied.to(target),
        )


def batch_statistics(
    batch: RingBudgetBatch,
    config: RingBudgetFeedConfig,
) -> dict[str, object]:
    if batch.frames.ndim != 5:
        raise ValueError("batch frames must be rank five")
    expected_frame_shape = (
        batch.frames.shape[0],
        config.sequence_length,
        *FRAME_SHAPE,
    )
    if tuple(batch.frames.shape) != expected_frame_shape:
        raise ValueError("batch frames do not match the feed configuration")
    expected_state_shape = (
        batch.frames.shape[0],
        config.sequence_length,
        config.position_count,
    )
    for name, value in (
        ("states", batch.states),
        ("labels", batch.labels),
        ("source_image_indices", batch.source_image_indices),
    ):
        if value.dtype != torch.int64 or tuple(value.shape) != expected_state_shape:
            raise ValueError(f"batch {name} do not match the feed configuration")
    expected_transition_shape = (
        batch.frames.shape[0],
        config.sequence_length - 1,
        config.position_count,
    )
    if (
        batch.copy_branch.dtype != torch.bool
        or tuple(batch.copy_branch.shape) != expected_transition_shape
    ):
        raise ValueError("batch copy branches do not match the feed configuration")
    for name, value in (
        ("random_proposals", batch.random_proposals),
        ("copied_targets", batch.copied_targets),
    ):
        if value.dtype != torch.int64 or tuple(value.shape) != expected_transition_shape:
            raise ValueError(f"batch {name} do not match the feed configuration")
    realized = batch.states[:, 1:] == batch.copied_targets
    result: dict[str, object] = {
        "groups": {},
        "positions": {},
        "class_histograms": {},
        "sampled_local_identity_count": int(torch.unique(batch.states).numel()),
        "sampled_source_identity_count": int(
            torch.unique(batch.source_image_indices).numel()
        ),
    }
    group_specs = (
        ("AC", config.ac_indices, config.q_ac),
        ("BD_ring", config.bd_indices, config.q_bd),
    )
    for group_name, indices, probability in group_specs:
        branch = batch.copy_branch[:, :, indices]
        random_branch = ~branch
        group_realized = realized[:, :, indices]
        result["groups"][group_name] = {
            "configured_copy_probability": probability,
            "configured_position_copy_probabilities": [
                config.probabilities[index] for index in indices
            ],
            "copy_branch_rate": float(branch.float().mean()),
            "realized_exact_identity_copy_rate": float(
                group_realized.float().mean()
            ),
            "random_branch_identity_match_rate": (
                float(group_realized[random_branch].float().mean())
                if bool(random_branch.any())
                else None
            ),
        }
    for index, (name, probability) in enumerate(
        zip(config.position_names, config.probabilities, strict=True)
    ):
        branch = batch.copy_branch[:, :, index]
        random_branch = ~branch
        position_realized = realized[:, :, index]
        result["positions"][name] = {
            "configured_copy_probability": probability,
            "copy_branch_rate": float(branch.float().mean()),
            "realized_exact_identity_copy_rate": float(
                position_realized.float().mean()
            ),
            "random_branch_identity_match_rate": (
                float(position_realized[random_branch].float().mean())
                if bool(random_branch.any())
                else None
            ),
        }
        result["class_histograms"][name] = torch.bincount(
            batch.labels[:, :, index].reshape(-1).cpu(),
            minlength=NUM_CLASSES,
        ).tolist()
    return result


def pool_contract(pool: PositionImagePool) -> dict[str, object]:
    if not isinstance(pool, PositionImagePool):
        raise TypeError("pool must be a ring_budget_switch PositionImagePool")
    return {
        "identity_count": pool.identity_count,
        "classes": list(pool.dataset_classes),
        "class_histogram": torch.bincount(
            pool.labels, minlength=NUM_CLASSES
        ).tolist(),
        "source_indices_sha256": source_indices_sha256(pool.source_indices),
        "minimum_source_index": int(pool.source_indices.min()),
        "maximum_source_index": int(pool.source_indices.max()),
    }


__all__ = [
    "BD_LAYOUTS",
    "AC_INDICES",
    "AC_POSITION_NAMES",
    "BD_POSITION_NAMES",
    "FRAME_SHAPE",
    "IDENTITY_COUNT",
    "SHARED_DRAW_WIDTH",
    "MAX_POSITION_COUNT",
    "VALID_BD_RING_SIZES",
    "PositionImagePool",
    "RingBudgetBatch",
    "RingBudgetFeed",
    "RingBudgetFeedConfig",
    "batch_statistics",
    "bd_tile_size",
    "bd_tile_coordinates_yx",
    "copied_targets_for_states",
    "pool_contract",
    "position_names",
    "render_collages",
    "ring_budget_feed_config_from_mapping",
    "sample_state_sequences",
    "select_ring_pool",
    "source_position_for_target",
]
