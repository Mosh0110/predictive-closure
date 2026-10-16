from __future__ import annotations
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
import torch
from torch import Tensor
from jepa.data import CIFAR10Pool
from .feed import (
    POSITION_COUNT,
    DictionaryFeedConfig,
    FourImageBatch,
    dictionary_sha256,
    identity_pool_sha256,
    mapped_targets_for_states,
    render_collages,
)


FORMAT_VERSION = 2
ARTIFACT_KIND = "jepa_four_image_dictionary_id_corpus"
MAX_UINT16_ID = 2**16 - 1


def tensor_sha256(tensor: Tensor) -> str:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("tensor hashing requires a contiguous CPU tensor")
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


class MaterializedSequenceCorpus:
    def __init__(self, artifact: Mapping[str, Any], path: Path) -> None:
        version = artifact.get("format_version")
        required = {
            "format_version",
            "artifact_kind",
            "feed",
            "state_count",
            "corpus_seed",
            "epoch_count",
            "sequences_per_epoch",
            "states",
            "dictionary_branch",
            "states_sha256",
            "dictionary_branch_sha256",
            "dictionary_sha256",
            "dataset_file_sha256",
            "identity_source_indices",
            "identity_source_indices_sha256",
        }
        if set(artifact) != required:
            raise ValueError(
                f"corpus fields differ: missing={sorted(required - set(artifact))}, "
                f"extra={sorted(set(artifact) - required)}"
            )
        if version != FORMAT_VERSION:
            raise ValueError("unsupported corpus format version")
        if artifact["artifact_kind"] != ARTIFACT_KIND:
            raise ValueError("artifact is not a four-image ID corpus")
        self.path = path
        self.feed_config = DictionaryFeedConfig(**artifact["feed"])
        self.state_count = int(artifact["state_count"])
        self.identity_source_indices = artifact["identity_source_indices"]
        self.corpus_seed = int(artifact["corpus_seed"])
        self.epoch_count = int(artifact["epoch_count"])
        self.sequences_per_epoch = int(artifact["sequences_per_epoch"])
        self.states = artifact["states"]
        self.dictionary_branch = artifact["dictionary_branch"]
        self.dictionary_sha256 = str(artifact["dictionary_sha256"])
        self.dataset_file_sha256 = dict(artifact["dataset_file_sha256"])
        expected_states = (
            self.epoch_count,
            self.sequences_per_epoch,
            self.feed_config.sequence_length,
            POSITION_COUNT,
        )
        expected_branch = (
            self.epoch_count,
            self.sequences_per_epoch,
            self.feed_config.sequence_length - 1,
            POSITION_COUNT,
        )
        if self.states.device.type != "cpu" or self.states.dtype != torch.uint16:
            raise TypeError("corpus states must be CPU uint16")
        if self.dictionary_branch.device.type != "cpu" or self.dictionary_branch.dtype != torch.bool:
            raise TypeError("corpus dictionary branches must be CPU bool")
        if tuple(self.states.shape) != expected_states:
            raise ValueError(f"corpus states must have shape {expected_states}")
        if tuple(self.dictionary_branch.shape) != expected_branch:
            raise ValueError(f"corpus branches must have shape {expected_branch}")
        if not self.states.is_contiguous() or not self.dictionary_branch.is_contiguous():
            raise ValueError("corpus tensors must be contiguous")
        if self.state_count < 2 or self.state_count > MAX_UINT16_ID + 1:
            raise ValueError("corpus state_count is incompatible with uint16 storage")
        if (
            self.identity_source_indices.device.type != "cpu"
            or self.identity_source_indices.dtype != torch.int64
            or self.identity_source_indices.ndim != 1
        ):
            raise TypeError("identity_source_indices must be a rank-one CPU int64 tensor")
        if self.identity_source_indices.numel() != self.state_count:
            raise ValueError("identity_source_indices has the wrong length")
        if self.identity_source_indices.unique().numel() != self.state_count:
            raise ValueError("identity_source_indices contains duplicates")
        if identity_pool_sha256(
            self.identity_source_indices
        ) != artifact["identity_source_indices_sha256"]:
            raise ValueError("identity source-index hash mismatch")
        if self.feed_config.identity_count != self.state_count:
            raise ValueError("feed identity_count differs from corpus state_count")
        if int(self.states.to(torch.int32).max()) >= self.state_count:
            raise ValueError("corpus contains an out-of-range image ID")
        if tensor_sha256(self.states) != artifact["states_sha256"]:
            raise ValueError("corpus state tensor hash mismatch")
        if tensor_sha256(self.dictionary_branch) != artifact["dictionary_branch_sha256"]:
            raise ValueError("corpus branch tensor hash mismatch")

    @classmethod
    def load(cls, path: str | Path) -> "MaterializedSequenceCorpus":
        resolved = Path(path).expanduser().resolve()
        artifact = torch.load(resolved, map_location="cpu", weights_only=False)
        if not isinstance(artifact, Mapping):
            raise TypeError("corpus artifact must be a mapping")
        return cls(artifact, resolved)

    def validate_against(
        self,
        *,
        pool: CIFAR10Pool,
        identity_source_indices: Tensor,
        feed_config: DictionaryFeedConfig,
        dictionaries: Tensor,
        dataset_file_sha256: Mapping[str, str],
        expected_seed: int,
    ) -> None:
        if self.feed_config != feed_config:
            raise ValueError("corpus feed configuration differs from the run")
        if self.state_count != pool.images.shape[0]:
            raise ValueError("corpus identity count differs from CIFAR pool")
        if not torch.equal(self.identity_source_indices, identity_source_indices):
            raise ValueError("corpus identity subset differs from the run")
        if self.dictionary_sha256 != dictionary_sha256(dictionaries):
            raise ValueError("corpus dictionary hash differs from the run")
        if self.dataset_file_sha256 != dict(dataset_file_sha256):
            raise ValueError("corpus CIFAR hashes differ from the run")
        if self.corpus_seed != expected_seed:
            raise ValueError("corpus seed differs from train_sequence_seed")

    def batch(
        self,
        epoch_index: int,
        start: int,
        stop: int,
        *,
        pool: CIFAR10Pool,
        dictionaries: Tensor,
        device: str | torch.device,
        render_layout: str = "columns",
    ) -> FourImageBatch:
        if not 0 <= epoch_index < self.epoch_count:
            raise IndexError("epoch index is out of range")
        if not 0 <= start < stop <= self.sequences_per_epoch:
            raise IndexError("corpus batch bounds are invalid")
        states = self.states[epoch_index, start:stop].to(torch.int64)
        branch = self.dictionary_branch[epoch_index, start:stop]
        mapped = mapped_targets_for_states(states, dictionaries)
        target = torch.device(device)
        images = pool.images[states].to(target, dtype=torch.float32).div_(255.0)
        frames = render_collages(images, layout=render_layout)
        return FourImageBatch(
            frames=frames,
            states=states.to(target),
            labels=pool.labels[states].to(target),
            dictionary_branch=branch.to(target),
            random_proposals=None,
            mapped_targets=mapped.to(target),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format_version": FORMAT_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "feed": asdict(self.feed_config),
            "state_count": self.state_count,
            "identity_source_indices_sha256": identity_pool_sha256(
                self.identity_source_indices
            ),
            "corpus_seed": self.corpus_seed,
            "epoch_count": self.epoch_count,
            "sequences_per_epoch": self.sequences_per_epoch,
            "states_sha256": tensor_sha256(self.states),
            "dictionary_branch_sha256": tensor_sha256(self.dictionary_branch),
            "dictionary_sha256": self.dictionary_sha256,
            "dataset_file_sha256": self.dataset_file_sha256,
            "storage": {
                "states_dtype": str(self.states.dtype),
                "branch_dtype": str(self.dictionary_branch.dtype),
                "states_bytes": self.states.numel() * self.states.element_size(),
                "branch_bytes": self.dictionary_branch.numel()
                * self.dictionary_branch.element_size(),
            },
        }


__all__ = [
    "ARTIFACT_KIND",
    "FORMAT_VERSION",
    "MaterializedSequenceCorpus",
    "tensor_sha256",
]
