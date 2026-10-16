from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
import torch
from torch import Tensor
from jepa.data import CIFAR10Pool, NUM_CLASSES, source_indices_sha256


POSITIONS = ("A", "C", "B", "D")
POSITION_COUNT = len(POSITIONS)
SOURCE_POSITION_FOR_TARGET = (2, 3, 0, 1)
FRAME_SHAPE = (3, 64, 64)
DICTIONARY_MODES = ("random_permutation", "identity")
RENDER_LAYOUTS = ("columns", "rows", "diagonals")
LOGICAL_INDEX_AT_PHYSICAL_SLOT = {
    "columns": (0, 1, 2, 3),
    "rows": (0, 2, 1, 3),
    "diagonals": (0, 1, 3, 2),
}


def _probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")


@dataclass(frozen=True)
class DictionaryFeedConfig:
    sequence_length: int = 8
    probability_a: float = 1.0
    probability_b: float = 0.9
    probability_c: float = 0.8
    probability_d: float = 0.3
    dictionary_seed: int = 20260804
    dictionary_mode: str = "random_permutation"
    identity_count: int = 50_000
    identity_subset_seed: int = 20260805

    def __post_init__(self) -> None:
        if type(self.sequence_length) is not int or self.sequence_length < 2:
            raise ValueError("sequence_length must be an integer of at least two")
        for name in ("probability_a", "probability_b", "probability_c", "probability_d"):
            _probability(name, getattr(self, name))
        if type(self.dictionary_seed) is not int or not 0 <= self.dictionary_seed < 2**63:
            raise ValueError("dictionary_seed must be an integer in [0,2^63)")
        if self.dictionary_mode not in DICTIONARY_MODES:
            raise ValueError(
                f"dictionary_mode must be one of {DICTIONARY_MODES}, "
                f"got {self.dictionary_mode!r}"
            )
        if (
            type(self.identity_count) is not int
            or self.identity_count < 2 * NUM_CLASSES
            or self.identity_count % NUM_CLASSES
        ):
            raise ValueError("identity_count must be divisible by 10 and at least 20")
        if (
            type(self.identity_subset_seed) is not int
            or not 0 <= self.identity_subset_seed < 2**63
        ):
            raise ValueError("identity_subset_seed must be an integer in [0,2^63)")

    @property
    def probabilities(self) -> tuple[float, float, float, float]:
        return (
            self.probability_a,
            self.probability_b,
            self.probability_c,
            self.probability_d,
        )


@dataclass(frozen=True)
class FourImageBatch:
    frames: Tensor
    states: Tensor
    labels: Tensor
    dictionary_branch: Tensor
    random_proposals: Tensor | None
    mapped_targets: Tensor


def select_identity_pool(
    full_pool: CIFAR10Pool,
    *,
    identity_count: int,
    seed: int,
) -> tuple[CIFAR10Pool, Tensor]:
    if type(identity_count) is not int or identity_count < 2 * NUM_CLASSES:
        raise ValueError("identity_count must be at least 20")
    if identity_count % NUM_CLASSES:
        raise ValueError("identity_count must be divisible by 10")
    if identity_count > full_pool.images.shape[0]:
        raise ValueError("identity_count exceeds the CIFAR pool")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2^63)")
    if identity_count == full_pool.images.shape[0]:
        source_indices = torch.arange(identity_count, dtype=torch.int64)
    else:
        per_class = identity_count // NUM_CLASSES
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = []
        for label in range(NUM_CLASSES):
            candidates = torch.nonzero(
                full_pool.labels == label, as_tuple=False
            ).squeeze(1)
            permutation = torch.randperm(candidates.numel(), generator=generator)
            selected.append(candidates[permutation[:per_class]])
        source_indices = torch.cat(selected).sort().values.contiguous()
    pool = CIFAR10Pool(
        images=full_pool.images.index_select(0, source_indices).contiguous(),
        labels=full_pool.labels.index_select(0, source_indices).contiguous(),
        split=f"{full_pool.split}_balanced_{identity_count}",
    )
    if pool.images.shape[0] != identity_count:
        raise AssertionError("identity subset has the wrong size")
    if source_indices.unique().numel() != identity_count:
        raise AssertionError("identity subset contains duplicates")
    return pool, source_indices


def identity_pool_sha256(source_indices: Tensor) -> str:
    return source_indices_sha256(source_indices)


def mapped_targets_for_states(states: Tensor, dictionaries: Tensor) -> Tensor:
    if states.ndim != 3 or states.shape[2] != POSITION_COUNT or states.shape[1] < 2:
        raise ValueError("states must have shape [batch,time,4] with time at least two")
    if states.dtype != torch.int64:
        raise TypeError("states must be int64")
    if dictionaries.device.type != "cpu" or dictionaries.dtype != torch.int64:
        raise TypeError("dictionaries must be CPU int64")
    source_positions = torch.tensor(SOURCE_POSITION_FOR_TARGET, dtype=torch.int64)
    previous_sources = states[:, :-1].cpu().index_select(2, source_positions)
    flattened = previous_sources.reshape(-1, POSITION_COUNT)
    mapped = dictionaries.gather(1, flattened.T).T
    return mapped.reshape(states.shape[0], states.shape[1] - 1, POSITION_COUNT)


def build_dictionaries(
    state_count: int,
    seed: int,
    *,
    mode: str = "random_permutation",
) -> Tensor:
    if type(state_count) is not int or state_count < 2:
        raise ValueError("state_count must be at least two")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2^63)")
    if mode not in DICTIONARY_MODES:
        raise ValueError(f"mode must be one of {DICTIONARY_MODES}, got {mode!r}")
    if mode == "identity":
        return torch.arange(state_count, dtype=torch.int64).repeat(POSITION_COUNT, 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dictionaries = torch.stack(
        [torch.randperm(state_count, generator=generator) for _ in POSITIONS]
    )
    expected = torch.arange(state_count)
    for dictionary in dictionaries:
        if not torch.equal(dictionary.sort().values, expected):
            raise AssertionError("a generated dictionary is not a permutation")
    for left in range(POSITION_COUNT):
        for right in range(left + 1, POSITION_COUNT):
            if torch.equal(dictionaries[left], dictionaries[right]):
                raise AssertionError("two independently generated dictionaries are equal")
    return dictionaries


def dictionary_sha256(dictionaries: Tensor) -> str:
    if dictionaries.device.type != "cpu" or dictionaries.dtype != torch.int64:
        raise TypeError("dictionaries must be CPU int64")
    if dictionaries.ndim != 2 or dictionaries.shape[0] != POSITION_COUNT:
        raise ValueError("dictionaries must have shape [4,state_count]")
    little_endian = dictionaries.contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def sample_state_sequences(
    batch_size: int,
    state_count: int,
    dictionaries: Tensor,
    config: DictionaryFeedConfig,
    *,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if generator.device.type != "cpu":
        raise ValueError("state generator must remain on CPU")
    if dictionaries.shape != (POSITION_COUNT, state_count):
        raise ValueError("dictionaries have the wrong shape")
    if dictionaries.device.type != "cpu" or dictionaries.dtype != torch.int64:
        raise TypeError("dictionaries must be CPU int64")
    probabilities = torch.tensor(config.probabilities, dtype=torch.float32)
    source_positions = torch.tensor(SOURCE_POSITION_FOR_TARGET, dtype=torch.int64)
    states = [
        torch.randint(
            state_count,
            (batch_size, POSITION_COUNT),
            generator=generator,
            dtype=torch.int64,
        )
    ]
    branch_masks = []
    random_proposals = []
    mapped_targets = []
    for _ in range(1, config.sequence_length):
        previous_sources = states[-1].index_select(1, source_positions)
        mapped = dictionaries.gather(1, previous_sources.T).T.contiguous()
        branch = torch.rand(
            batch_size, POSITION_COUNT, generator=generator
        ) < probabilities
        random = torch.randint(
            state_count,
            (batch_size, POSITION_COUNT),
            generator=generator,
            dtype=torch.int64,
        )
        states.append(torch.where(branch, mapped, random))
        branch_masks.append(branch)
        random_proposals.append(random)
        mapped_targets.append(mapped)
    return (
        torch.stack(states, dim=1),
        torch.stack(branch_masks, dim=1),
        torch.stack(random_proposals, dim=1),
        torch.stack(mapped_targets, dim=1),
    )


def render_collages(images: Tensor, *, layout: str = "columns") -> Tensor:
    if images.ndim != 6 or tuple(images.shape[2:]) != (4, 3, 32, 32):
        raise ValueError("images must have shape [batch,time,4,3,32,32]")
    if not images.is_floating_point():
        raise TypeError("images must be floating point")
    if layout not in RENDER_LAYOUTS:
        raise ValueError(f"layout must be one of {RENDER_LAYOUTS}, got {layout!r}")
    batch_size, time_steps = images.shape[:2]
    frames = torch.zeros(
        batch_size,
        time_steps,
        *FRAME_SHAPE,
        dtype=images.dtype,
        device=images.device,
    )
    logical_at_physical = LOGICAL_INDEX_AT_PHYSICAL_SLOT[layout]
    frames[:, :, :, :32, :32] = images[:, :, logical_at_physical[0]]
    frames[:, :, :, :32, 32:] = images[:, :, logical_at_physical[1]]
    frames[:, :, :, 32:, :32] = images[:, :, logical_at_physical[2]]
    frames[:, :, :, 32:, 32:] = images[:, :, logical_at_physical[3]]
    return frames


def transition_statistics(
    batch: FourImageBatch,
) -> dict[str, dict[str, float | None]]:
    realized = batch.states[:, 1:] == batch.mapped_targets
    result = {}
    for index, position in enumerate(POSITIONS):
        dictionary_branch = batch.dictionary_branch[:, :, index]
        random_branch = ~dictionary_branch
        result[position] = {
            "dictionary_branch_rate": float(dictionary_branch.float().mean()),
            "realized_dictionary_match_rate": float(realized[:, :, index].float().mean()),
            "random_branch_dictionary_match_rate": (
                float(realized[:, :, index][random_branch].float().mean())
                if bool(random_branch.any())
                else None
            ),
        }
    return result


class FourImageDictionaryFeed:
    def __init__(self, pool: CIFAR10Pool, config: DictionaryFeedConfig) -> None:
        if pool.images.shape[0] != config.identity_count:
            raise ValueError(
                "pool identity count differs from the feed configuration: "
                f"{pool.images.shape[0]} != {config.identity_count}"
            )
        self.pool = pool
        self.config = config
        self.dictionaries = build_dictionaries(
            pool.images.shape[0],
            config.dictionary_seed,
            mode=config.dictionary_mode,
        )

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: str | torch.device,
        render_layout: str = "columns",
    ) -> FourImageBatch:
        states, branch, random, mapped = sample_state_sequences(
            batch_size,
            self.pool.images.shape[0],
            self.dictionaries,
            self.config,
            generator=generator,
        )
        target = torch.device(device)
        images = self.pool.images[states].to(target, dtype=torch.float32).div_(255.0)
        frames = render_collages(images, layout=render_layout)
        if tuple(frames.shape) != (
            batch_size,
            self.config.sequence_length,
            *FRAME_SHAPE,
        ):
            raise AssertionError("renderer produced the wrong shape")
        return FourImageBatch(
            frames=frames,
            states=states.to(target),
            labels=self.pool.labels[states].to(target),
            dictionary_branch=branch.to(target),
            random_proposals=random.to(target),
            mapped_targets=mapped.to(target),
        )


__all__ = [
    "FRAME_SHAPE",
    "DICTIONARY_MODES",
    "RENDER_LAYOUTS",
    "LOGICAL_INDEX_AT_PHYSICAL_SLOT",
    "POSITIONS",
    "SOURCE_POSITION_FOR_TARGET",
    "DictionaryFeedConfig",
    "FourImageBatch",
    "FourImageDictionaryFeed",
    "build_dictionaries",
    "dictionary_sha256",
    "mapped_targets_for_states",
    "render_collages",
    "sample_state_sequences",
    "select_identity_pool",
    "identity_pool_sha256",
    "transition_statistics",
]
